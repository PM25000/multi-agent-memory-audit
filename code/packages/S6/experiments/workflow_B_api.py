"""Bounded native function calling for the external-workflow experiment.

Responses tools for the authorized airouting GPT; Chat tools for OpenRouter.
No custom decision wrapper, no hidden SDK retry, no backend actions in this file.
Documentation: https://developers.openai.com/api/docs/guides/function-calling
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

from review_fair_comparison_ids import expected_model_spec
from review_revision import RevisionCalls


def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    t.replace(p)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def build_request(spec, history, schemas, max_tokens=2400):
    if spec.get("api") == "responses":
        definitions = [{"type": "function", **deepcopy(t["function"]), "strict": False} for t in schemas]
        request = {"model": spec["model"], "input": history, "tools": definitions,
                "tool_choice": "auto", "parallel_tool_calls": False, "store": False,
                "reasoning": {"effort": "none"}, "max_output_tokens": max_tokens}
    else:
        request = {"model": spec["model"], "messages": history, "tools": schemas,
                   "tool_choice": "auto", "parallel_tool_calls": False,
                   "temperature": 0, "max_tokens": max_tokens,
                   "extra_body": {"reasoning": {"enabled": False}}}
    if not schemas:
        for key in ("tools", "tool_choice", "parallel_tool_calls"):
            request.pop(key)
    return request


def parse_wire_body(text, responses):
    """Accept JSON or a complete Responses SSE envelope; never synthesize output."""
    try:
        raw = json.loads(text)
        return raw, "json"
    except ValueError:
        if not responses:
            raise
    # Some compatible gateways return event framing even for nonstream requests.
    # Only a complete terminal response is accepted, never pieced-together text.
    terminal = []
    for line in text.splitlines():
        if line.startswith("data:"):
            content = line[5:].strip()
            if content == "[DONE]":
                continue
            try:
                event = json.loads(content)
            except ValueError:
                continue
            if event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                if isinstance(event.get("response"), dict):
                    terminal.append(event["response"])
    if len(terminal) != 1:
        raise ValueError("missing_or_multiple_terminal_responses")
    return terminal[0], "sse_terminal_response"


def response_items(raw, responses):
    functions, text = [], []
    if responses:
        completed = raw.get("status") == "completed"
        history = raw.get("output") or []
        for item in history:
            if item.get("type") == "function_call":
                functions.append({"id": item.get("call_id"), "name": item.get("name"), "arguments": item.get("arguments")})
            elif item.get("type") == "message":
                text.extend(x.get("text", "") for x in item.get("content", []) if x.get("type") == "output_text")
    else:
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        completed = choice.get("finish_reason") in ("stop", "tool_calls")
        history = [{k: msg[k] for k in ("role", "content", "tool_calls") if k in msg}]
        text = [msg.get("content") or ""]
        functions = [{"id": x.get("id"), "name": x.get("function", {}).get("name"), "arguments": x.get("function", {}).get("arguments")}
                     for x in msg.get("tool_calls") or []]
    errors = []
    if len(functions) > 1:
        errors.append("parallel_calls_despite_disabled")
    for f in functions:
        try:
            f["args"] = json.loads(f["arguments"])
            if not isinstance(f["args"], dict) or not f["id"] or not f["name"]:
                raise ValueError("invalid_function_shape")
        except (TypeError, ValueError):
            errors.append("invalid_tool_arguments")
            f["args"] = None
    return {"complete": completed, "functions": functions, "text": "".join(text),
            "history_items": history, "format_errors": errors}


def append_result(history, output, call, result, responses):
    history.extend(deepcopy(output["history_items"]))
    encoded = json.dumps(result, ensure_ascii=False, default=str)
    if responses:
        history.append({"type": "function_call_output", "call_id": call["id"], "output": encoded})
    else:
        history.append({"role": "tool", "tool_call_id": call["id"], "content": encoded})


class NativeCalls:
    def __init__(self, root, model):
        assert model in ("qwen", "glm", "gpt55_responses")
        self.root, self.model = Path(root), model
        self.spec = expected_model_spec(model)
        subject = "gpt" if model == "gpt55_responses" else model
        self.holder = RevisionCalls(self.root / "client", model=subject, budget_cny=100000)
        assert self.holder.specs["gpt"] == self.spec
        self.client = self.holder.clients["gpt"]
        assert self.client.max_retries == 0
        self.responses = self.spec.get("api") == "responses"
        if self.responses:
            assert self.spec["model"] == "gpt-5.5" and "airouting" in self.spec["base_url"]
        self.guard = threading.Lock()
        self.locks = {}
        self.stop = threading.Event()
        self.consecutive_failures = 0
        self.failed_identities_since_success = set()
        write(self.root / "model_config.json", {"spec": self.spec, "sdk_retries": 0,
            "transport_attempts": 3, "tool_choice": "auto", "parallel_tool_calls": False,
            "response_parser": "JSON or a complete terminal SSE envelope; raw body retained"})

    def one_attempt(self, request, phase, attempt):
        identity = digest({"phase": phase, "endpoint": self.spec["base_url"], "request": request})
        key = digest({"identity": identity, "attempt": attempt})
        path = self.root / "wire" / f"{key}.json"
        if path.exists():
            return read(path), True
        row = {"request_identity": identity, "request_hash": key, "phase": phase,
               "attempt": attempt, "request": request, "endpoint": self.spec["base_url"],
               "started_utc": datetime.now(timezone.utc).isoformat(), "status": None,
               "raw_body": None, "raw_response": None, "usage": {}, "parsed": None}
        start = time.perf_counter()
        try:
            response = (self.client.responses.with_raw_response.create(**request) if self.responses
                        else self.client.chat.completions.with_raw_response.create(**request))
            body = response.text
            # Avoid storing a hypothetical credential echo from a broken gateway.
            key_value = self.client.api_key
            if isinstance(key_value, str) and key_value:
                body = body.replace(key_value, "[REDACTED_CREDENTIAL]")
            row["raw_body"] = body
            row["http_status"] = response.status_code
            row["content_type"] = response.headers.get("content-type")
            raw, framing = parse_wire_body(body, self.responses)
            row.update(raw_response=raw, framing=framing, usage=raw.get("usage") or {})
            parsed = response_items(raw, self.responses)
            row.update(parsed=parsed, status="ok" if parsed["complete"] else "incomplete_response")
        except Exception as exc:
            row.update(status="transport_failure", error_type=type(exc).__name__,
                       exception_http_status=getattr(exc, "status_code", None))
        row.update(wall_seconds=time.perf_counter() - start, finished_utc=datetime.now(timezone.utc).isoformat())
        write(path, row)
        return row, False

    def call(self, history, schemas, phase, max_tokens=2400):
        request = build_request(self.spec, history, schemas, max_tokens)
        identity = digest({"phase": phase, "endpoint": self.spec["base_url"], "request": request})
        with self.guard:
            lock = self.locks.setdefault(identity, threading.Lock())
        with lock:
            if self.stop.is_set():
                return {"status": "not_attempted_batch_stopped", "trace": [], "parsed": None}
            trace = []
            for attempt in range(1, 4):
                row, cached = self.one_attempt(request, phase, attempt)
                trace.append({"request_hash": row["request_hash"], "status": row["status"], "cache_hit": cached})
                if row["status"] != "transport_failure":
                    with self.guard:
                        self.consecutive_failures = 0
                        self.failed_identities_since_success.clear()
                    break
                if row.get("exception_http_status") in (401, 403):
                    self.stop.set()
                    break
                if attempt < 3:
                    time.sleep(2 * attempt)
            if row["status"] == "transport_failure":
                with self.guard:
                    self.failed_identities_since_success.add(identity)
                    self.consecutive_failures = len(self.failed_identities_since_success)
                    if self.consecutive_failures >= 3:
                        self.stop.set()
            return {"status": row["status"], "trace": trace, "parsed": row["parsed"], "request_identity": identity}
