"""Persistent one-attempt native Chat client for a new same-input experiment.

Importing this module never reads credentials, writes files or calls an API.
Only call() may send. No SDK/transport retry, format repair or model fallback.
The fixed B parser and history adapter remain the interpretation boundary.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

from workflow_B_api import append_result, build_request, parse_wire_body, response_items

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "glm": {"model": "z-ai/glm-5.2", "aliases": ["z-ai/glm-5.2", "glm-5.2"]},
    "qwen": {"model": "qwen/qwen3.6-27b", "aliases": ["qwen/qwen3.6-27b", "qwen3.6-27b"]},
}
ENDPOINT = "https://openrouter.ai/api/v1"


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_exclusive(path, value):
    """Durable complete JSON then atomic hard-link publication; never overwrite.

    A failed publication is an error, not an instruction to resend a request.
    A crash after intent publication permanently consumes that logical ID.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


class ScopeStopped(RuntimeError):
    pass


class IntentOnly(RuntimeError):
    pass


@contextmanager
def scope_lock(root):
    """Nonblocking OS lock around cap check, intent publication and dispatch."""
    path = Path(root) / "dispatch.lock"
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ScopeStopped("scope_in_use_no_request") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ScopeStopped("scope_in_use_no_request") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


class NativeClient:
    """One sequential model scope. A root must belong to this client/version.

    request_cap is a persistent intent count, including failed requests and
    crash-ambiguous intents. Cached receipts consume no new API budget. An
    optional injected client exists for offline checks; no fake outcomes are
    used in a live experiment. Defaults/limits are recorded, never tuned here.
    """
    def __init__(self, root, model, *, request_cap, timeout_seconds=120,
                 max_tokens=2400, max_request_bytes=131072, client=None):
        if model not in SPECS:
            raise ValueError("Only explicitly configured glm/qwen are supported")
        if type(request_cap) is not int or request_cap <= 0:
            raise ValueError("request_cap must be a fixed positive integer")
        if not 0 < timeout_seconds <= 300 or type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("invalid timeout/output limit")
        if type(max_request_bytes) is not int or max_request_bytes <= 0:
            raise ValueError("invalid request byte limit")
        self.root, self.model = Path(root).resolve(), model
        self.spec = {"model": SPECS[model]["model"], "base_url": ENDPOINT, "key": "OPENROUTER_API_KEY"}
        self.responses = False
        self.cap, self.max_tokens, self.max_request_bytes = request_cap, max_tokens, max_request_bytes
        self.timeout_seconds = timeout_seconds
        self.lock = threading.Lock()
        self.secret = None
        self.client = client
        self.config = {"version": "same-input-native-chat-1", "model_key": model,
            "model": self.spec["model"], "endpoint": ENDPOINT,
            "accepted_response_models": SPECS[model]["aliases"], "request_cap": request_cap,
            "timeout_seconds": timeout_seconds, "max_output_tokens": max_tokens,
            "max_request_bytes": max_request_bytes, "sdk_retries": 0, "transport_attempts": 1,
            "temperature": 0, "reasoning_enabled": False,
            "parallel_tool_calls": False, "identity_mismatch_halts_entire_model_scope": True,
            "injected_offline_transport": client is not None}
        config_path = self.root / "config.json"
        if config_path.exists():
            if read(config_path) != self.config:
                raise ValueError("scope_configuration_drift")
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError("new_scope_must_not_reuse_unrelated_directory")
            atomic_exclusive(config_path, self.config)

    def _get_client(self):
        # Defer credential access/client creation until an actual eligible call.
        if self.client is None:
            from dotenv import dotenv_values
            from openai import OpenAI
            self.secret = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(ROOT / ".env").get("OPENROUTER_API_KEY")
            if not self.secret:
                raise RuntimeError("missing_OPENROUTER_API_KEY")
            self.client = OpenAI(api_key=self.secret, base_url=ENDPOINT,
                                 max_retries=0, timeout=self.timeout_seconds)
        if getattr(self.client, "max_retries", 0) != 0:
            raise ValueError("SDK_retry_must_be_zero")
        return self.client

    def _redact(self, text):
        secret = self.secret or getattr(self.client, "api_key", None)
        return text.replace(secret, "[REDACTED_CREDENTIAL]") if isinstance(secret, str) and secret else text

    def _halt(self, row):
        marker = self.root / "scope_halted.json"
        if not marker.exists():
            atomic_exclusive(marker, {"logical_id": row["logical_id"], "status": row["status"],
                "request_sha256": row["request_sha256"], "returned_model": row.get("returned_model")})

    def call(self, history, schemas, phase, *, logical_id, max_tokens=None):
        if not isinstance(logical_id, str) or not logical_id or not isinstance(phase, str) or not phase:
            raise ValueError("Explicit nonempty logical_id and phase are required")
        limit = self.max_tokens if max_tokens is None else max_tokens
        if type(limit) is not int or not 0 < limit <= self.max_tokens:
            raise ValueError("per-call output limit exceeds frozen scope limit")
        request = build_request(self.spec, deepcopy(history), deepcopy(schemas), max_tokens=limit)
        byte_count = len(encoded(request))
        key = hashlib.sha256(logical_id.encode("utf-8")).hexdigest()
        intent_path = self.root / "intents" / (key + ".json")
        response_path = self.root / "responses" / (key + ".json")
        rejection_path = self.root / "preflight_rejections" / (key + ".json")
        intended = {"logical_id": logical_id, "phase": phase, "model_key": self.model,
            "request": request, "request_sha256": digest(request), "request_bytes": byte_count,
            "scope_configuration_sha256": digest(self.config)}
        with self.lock, scope_lock(self.root):
            if rejection_path.exists():
                saved = read(rejection_path)
                if {k: saved[k] for k in intended} != intended:
                    raise ValueError("logical_id_reused_after_preflight_rejection")
                return saved | {"cache_hit": True}
            if intent_path.exists():
                old = read(intent_path)
                if {k: old[k] for k in intended} != intended:
                    raise ValueError("logical_id_reused_with_different_request")
                if not response_path.exists():
                    raise IntentOnly("intent_without_receipt_never_resend")
                saved = read(response_path)
                if saved["request_sha256"] != intended["request_sha256"] or saved["logical_id"] != logical_id:
                    raise ValueError("response_intent_link_mismatch")
                if saved["status"] in ("model_identity_mismatch", "model_identity_missing"):
                    self._halt(saved)
                return saved | {"cache_hit": True}
            if response_path.exists():
                raise ValueError("orphan_response")
            if (self.root / "scope_halted.json").exists():
                raise ScopeStopped("model_scope_halted_on_response_identity")
            if len(list((self.root / "intents").glob("*.json"))) >= self.cap:
                raise ScopeStopped("persistent_request_cap_reached")
            if byte_count > self.max_request_bytes:
                # No request intent or API consumption for a preflight rejection.
                rejected = intended | {"status": "input_cap_exceeded", "parsed": None,
                    "usage": {}, "api_attempted": False, "cache_hit": False}
                atomic_exclusive(rejection_path, rejected)
                return rejected
            intended["intent_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_exclusive(intent_path, intended)
            row = {"logical_id": logical_id, "phase": phase, "model_key": self.model,
                "request_sha256": intended["request_sha256"], "intent_utc": intended["intent_utc"],
                "status": None, "api_attempted": False, "raw_body": None, "raw_response": None,
                "usage": {}, "parsed": None, "cache_hit": False}
            started = time.perf_counter()
            try:
                client = self._get_client()
                row["api_attempted"] = True
                response = client.chat.completions.with_raw_response.create(**request)
                row["http_status"] = response.status_code
                row["raw_body"] = self._redact(response.text)
                raw, framing = parse_wire_body(row["raw_body"], responses=False)
                if not isinstance(raw, dict):
                    raise ValueError("response_envelope_not_object")
                row.update(raw_response=raw, framing=framing, usage=raw.get("usage") or {}, returned_model=raw.get("model"))
                if not raw.get("model"):
                    row["status"] = "model_identity_missing"
                elif raw["model"] not in SPECS[self.model]["aliases"]:
                    row["status"] = "model_identity_mismatch"
                else:
                    # The legacy parser does not gate the Chat role/choice
                    # envelope. Validate these transport facts before reusing
                    # its unchanged native tool/text interpretation.
                    choices = raw.get("choices")
                    if not isinstance(choices, list) or len(choices) != 1:
                        raise ValueError("exactly_one_native_choice_required")
                    message = choices[0].get("message") if isinstance(choices[0], dict) else None
                    if not isinstance(message, dict) or message.get("role") != "assistant":
                        raise ValueError("native_response_role_must_be_assistant")
                    parsed = response_items(raw, responses=False)
                    row["parsed"] = parsed
                    row["status"] = ("incomplete_response" if not parsed["complete"] else
                        "invalid_native_tool_response" if parsed["format_errors"] else "ok")
            except Exception as exc:
                # No exception message/credential text; retain precise type/status.
                row.update(status="transport_failure" if row["raw_body"] is None and row["api_attempted"] else
                           "client_preflight_failure" if not row["api_attempted"] else "invalid_response",
                           error_type=type(exc).__name__, exception_http_status=getattr(exc, "status_code", None))
            row["wall_seconds"] = time.perf_counter() - started
            row["completed_utc"] = datetime.now(timezone.utc).isoformat()
            # Persist the identity stop before receipt so a crash cannot allow
            # another logical ID after a detected wrong-model response.
            if row["status"] in ("model_identity_mismatch", "model_identity_missing"):
                self._halt(row)
            atomic_exclusive(response_path, row)
            return row


__all__ = ["NativeClient", "ScopeStopped", "IntentOnly", "append_result"]
