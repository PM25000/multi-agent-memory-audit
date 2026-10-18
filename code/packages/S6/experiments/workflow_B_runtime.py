"""Fixed AgentDojo runtime for Experiment B; no language-model calls.

Run with .venv-workflow-B/Scripts/python.exe. The upstream source is unmodified.
All tool effects are restricted to the in-memory benchmark environment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import locale
from threading import local
from typing import Any

# Upstream YAML loaders use the process default encoding. On Windows a GBK
# default can silently alter non-ASCII source text and frozen state hashes.
if locale.getpreferredencoding(False).lower().replace('-', '') != 'utf8':
    raise RuntimeError('Workflow B requires UTF-8 mode: run python -X utf8 (or PYTHONUTF8=1). Do not regenerate frozen hashes under another encoding.')

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third-party/agentdojo-089ed468"
COMMIT = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
VERSION = "v1.2.2"
RESULTS = ROOT / "experiments/results/review_revision_20260910/workflow_B_runtime_20260915"
sys.path.insert(0, str(SOURCE / "src"))

from pydantic import BaseModel  # noqa: E402
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite, get_suites  # noqa: E402
from agentdojo.types import text_content_block_from_string  # noqa: E402

_guard = local()


def _audit(event, args):
    if not getattr(_guard, "active", False):
        return
    if event.startswith(("socket.", "subprocess.", "os.spawn")) or event in {"os.system", "os.startfile"}:
        raise PermissionError(f"Benchmark tool external operation blocked: {event}")
    if event == "open":
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & 0x703):
            raise PermissionError("Benchmark tool filesystem write blocked")


sys.addaudithook(_audit)


@contextmanager
def sandbox_scope():
    """Thread-local network/process/write guard; model API calls outside are unaffected."""
    was_active = getattr(_guard, "active", False)
    _guard.active = True
    try:
        yield
    finally:
        _guard.active = was_active


def jsonable(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def snapshot(env):
    return jsonable(env)


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@dataclass
class LoadedTask:
    suite: Any
    task: Any
    pre_environment: Any
    env: Any
    runtime: FunctionsRuntime

    def tools_schema(self, function_names=None):
        return tools_schema(self.runtime, function_names)

    def call(self, name, args, call_id=None):
        return call_tool(self.env, self.runtime, name, args, call_id)


def load_task(suite_name: str, task_id: str, version: str = VERSION) -> LoadedTask:
    suite = get_suite(version, suite_name)
    task = suite.user_tasks[task_id]
    env = suite.load_and_inject_default_environment({})
    env = task.init_environment(env)
    return LoadedTask(suite, task, env.model_copy(deep=True), env, FunctionsRuntime(suite.tools))


def get_ground_truth(loaded: LoadedTask):
    """Oracle only: never include these arguments in model prompts."""
    return loaded.task.ground_truth(loaded.pre_environment.model_copy(deep=True))


def tools_schema(runtime_or_suite, function_names=None):
    """Unaltered native descriptions and Pydantic schemas; Chat API envelope."""
    functions = (runtime_or_suite.functions if isinstance(runtime_or_suite, FunctionsRuntime)
                 else {f.name: f for f in runtime_or_suite.tools})
    names = list(functions) if function_names is None else list(function_names)
    return [{"type": "function", "function": {"name": n, "description": functions[n].description,
             "parameters": functions[n].parameters.model_json_schema()}} for n in names]


def call_tool(env, runtime, name: str, args: dict, call_id=None):
    """Mutates env in place. Logs failed attempts and before/after state hashes."""
    if name in runtime.functions:
        fn = runtime.functions[name].run
        p = Path(inspect.getfile(fn)).resolve()
        if not p.is_relative_to(SOURCE / "src/agentdojo/default_suites/v1/tools"):
            raise ValueError(f"Not a fixed benchmark tool: {name}")
    trace = FunctionCall(function=name, args=args, id=call_id)
    before = digest(env)
    with sandbox_scope():
        result, error = runtime.run_function(env, name, args, raise_on_error=False)
    return {"function": name, "args": jsonable(args), "id": call_id,
            "result": jsonable(result), "error": error, "trace": trace.model_dump(mode="json"),
            "before_sha256": before, "after_sha256": digest(env)}


def native_utility(loaded, traces, model_output="", post_environment=None):
    """Use the suite's real precedence: trace-based utility, then state utility."""
    calls = [t if isinstance(t, FunctionCall) else FunctionCall.model_validate(t.get("trace", t)) for t in traces]
    post = loaded.env if post_environment is None else post_environment
    with sandbox_scope():
        value = loaded.suite._check_user_task_utility(
            loaded.task, [text_content_block_from_string(model_output)], loaded.pre_environment, post, calls)
    return bool(value)


def run_ground_truth(suite_name, task_id, version=VERSION):
    loaded = load_task(suite_name, task_id, version)
    traces = [loaded.call(c.function, dict(c.args), c.id) for c in get_ground_truth(loaded)]
    errors = [t["error"] for t in traces if t["error"]]
    try:
        utility = native_utility(loaded, traces, loaded.task.GROUND_TRUTH_OUTPUT)
        utility_error = None
    except Exception as exc:
        utility, utility_error = False, f"{type(exc).__name__}: {exc}"
    return {"suite": suite_name, "task_id": task_id, "version": version,
            "prompt": loaded.task.PROMPT, "ground_truth_output": loaded.task.GROUND_TRUTH_OUTPUT,
            "pre_environment": snapshot(loaded.pre_environment), "post_environment": snapshot(loaded.env),
            "traces": traces, "tool_errors": errors, "native_utility": utility,
            "utility_error": utility_error, "passed": not errors and utility}


def _method_origin(task, method):
    fn = getattr(type(task), method)
    return {"module": fn.__module__, "qualname": fn.__qualname__,
            "source": str(Path(inspect.getfile(fn)).resolve().relative_to(SOURCE)).replace("\\", "/"),
            "line": inspect.getsourcelines(fn)[1]}


def task_metadata(task):
    nonlocal_tasks = inspect.getclosurevars(task.ground_truth).nonlocals
    roots = [v for v in nonlocal_tasks.values() if hasattr(v, "PROMPT") and hasattr(v, "ground_truth")]
    return {"id": task.ID, "module": type(task).__module__, "qualname": type(task).__qualname__,
            "class_name": type(task).__name__, "prompt": task.PROMPT,
            "ground_truth": _method_origin(task, "ground_truth"),
            "utility": _method_origin(task, "utility"),
            "utility_from_traces": _method_origin(task, "utility_from_traces"),
            "bound_component_tasks": [task_metadata(v) for v in roots]}


def source_check():
    manifest = json.loads((SOURCE / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == COMMIT
    for row in manifest["files"]:
        data = (SOURCE / row["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == row["git_blob_sha"]
    return {"commit": COMMIT, "verified_files": len(manifest["files"]), "source_unchanged": True}


def build_inventory():
    RESULTS.mkdir(parents=True, exist_ok=True)
    source = source_check()
    rows = []
    for suite_name in ("workspace", "slack"):
        suite = get_suite(VERSION, suite_name)
        for task in suite.user_tasks.values():
            row = {"suite": suite_name, **task_metadata(task)}
            try:
                result = run_ground_truth(suite_name, task.ID)
                row.update({"oracle_passed": result["passed"], "native_utility": result["native_utility"],
                            "tool_errors": result["tool_errors"], "utility_error": result["utility_error"],
                            "ground_truth_functions": [t["function"] for t in result["traces"]]})
                path = RESULTS / "oracle" / suite_name / (task.ID + ".json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception as exc:
                row.update({"oracle_passed": False, "runtime_error": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
    result = {"source": source, "benchmark_version": VERSION, "task_metadata": rows,
              "python": sys.version, "python_executable": sys.executable,
              "dependencies": {n: importlib.metadata.version(n) for n in
                  ["pydantic", "PyYAML", "deepdiff", "docstring-parser", "cohere", "google-genai"]},
              "model_calls": 0}
    (RESULTS / "runtime_inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(rows), "oracle_passed": sum(r["oracle_passed"] for r in rows),
                      "failed": [r["suite"] + "/" + r["id"] for r in rows if not r["oracle_passed"]],
                      "inventory": str(RESULTS / "runtime_inventory.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inventory", "check"])
    args = parser.parse_args()
    if args.action == "inventory":
        build_inventory()
    else:
        print(json.dumps(source_check()))
