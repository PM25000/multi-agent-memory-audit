"""Format-only pilot amendment, preserving all v1 code and raw responses."""
from __future__ import annotations
import hashlib
import re
import shutil

import run_workflow_conditions as core
from workflow_conditions_business import parse_response as strict_parse

ORIGINAL_OUT = core.OUT
core.OUT = ORIGINAL_OUT.with_name(ORIGINAL_OUT.name + "_v2")
core.SOURCE_FILES += ["experiments/run_workflow_conditions_v2.py",
    "docs/paper/workflow-condition-format-amendment-20260915.md"]


def parse_with_fence_adapter(content):
    if isinstance(content, str):
        match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", content.strip())
        if match:
            content = match.group(1)
    return strict_parse(content)


core.parse_response = parse_with_fence_adapter
original_prepare = core.prepare
original_analyze = core.analyze


def prepare():
    if (core.OUT / "freeze.json").exists():
        f = core.check_freeze()
        assert f["version"] == "A-executable-v2-fence-adapter"
        return f
    source_summary = core.read(ORIGINAL_OUT / "pilot/glm/summary.json")
    assert source_summary["complete"] and not source_summary["pilot_gate"]["passed"]
    f = original_prepare()
    f["version"] = "A-executable-v2-fence-adapter"
    f["parent_freeze_hash"] = core.digest(core.read(ORIGINAL_OUT / "freeze.json"))
    f["amendment"] = "strip one complete external Markdown JSON fence; same prompt/semantics; regrade pilot before formal"
    core.write(core.OUT / "freeze.json", f)
    inherited = {}
    for p in sorted((ORIGINAL_OUT / "pilot/glm/wire").glob("*.json")):
        dest = core.OUT / "pilot/glm/wire" / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        inherited[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == inherited[p.name]
    core.write(core.OUT / "pilot/inherited_wire_manifest.json", {
        "source": str((ORIGINAL_OUT / "pilot/glm/wire").relative_to(core.ROOT)),
        "files": inherited, "new_model_calls": 0})
    good = '{"tool":"t","decision":"execute","args":{},"missing_fields":[]}'
    assert parse_with_fence_adapter(good) == strict_parse(good)
    assert parse_with_fence_adapter("```json\n" + good + "\n```") == strict_parse(good)
    assert parse_with_fence_adapter("```\n" + good + "\n```") == strict_parse(good)
    for bad in ("Text\n```json\n" + good + "\n```", "```json\n" + good + "\n```\nText", "```json\n{}\n```", "```json\n[\n```", "```json\n" + good + "\n```\n```json\n" + good + "\n```"):
        assert parse_with_fence_adapter(bad) is None
    core.write(core.OUT / "format_adapter_validation.json", {"passed": True, "cases": 8})
    return f


def analyze(phase, model, audit=True):
    summary = original_analyze(phase, model, audit)
    rows = [core.read(p) for p in sorted((core.OUT / phase / model / "rows").glob("*.json"))]
    summary["strict_unadapted_format_valid"] = sum(strict_parse(n["content"]) is not None for r in rows for n in r["nodes"] if n["status"] == "ok")
    if phase == "pilot":
        inherited = core.read(core.OUT / "pilot/inherited_wire_manifest.json")
        for name, sha in inherited["files"].items():
            assert hashlib.sha256((core.OUT / "pilot/glm/wire" / name).read_bytes()).hexdigest() == sha
        summary["inherited_v1_wire_attempts"] = len(inherited["files"])
        summary["new_wire_attempts_this_version"] = summary["actual_wire_attempts"] - len(inherited["files"])
        assert summary["new_wire_attempts_this_version"] == 0, "pilot_amendment_must_be_offline"
    else:
        summary["new_wire_attempts_this_version"] = summary["actual_wire_attempts"]
    core.write(core.OUT / phase / model / "summary.json", summary)
    return summary


core.prepare = prepare
core.analyze = analyze

if __name__ == "__main__":
    core.main()
