"""Experiment A: freeze synthetic fixtures, execute bounded calls, audit offline.

Usage: prepare; run --phase pilot --model glm; analyze --phase pilot;
       run --phase formal --model glm|gpt55_responses; analyze; check
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

from workflow_conditions_business import (ROOT, DESIGN, SYSTEM, digest, semantic_case,
    payload, parse_response, score_workflow, validate_semantics)

OUT = ROOT / "experiments/results/review_revision_20260910/workflow_conditions_A_20260915"
MODELS = ("glm", "gpt55_responses")
LEVELS = ("low", "medium", "high")
PATTERNS = ("synchronized_slots", "staggered_slots", "no_omission")
SOURCE_FILES = ["experiments/run_workflow_conditions.py", "experiments/workflow_conditions_business.py",
    "experiments/sandbox_review_deployment.py", "experiments/review_fair_comparison_ids.py",
    "experiments/review_revision.py", "experiments/gpt_review_pilot.py", "experiments/active_models.json",
    "docs/paper/workflow-condition-execution-protocol-20260915.md",
    "docs/paper/workflow-condition-protocol-20260915.md",
    str((DESIGN / "controlled_structural_cases.json").relative_to(ROOT))]


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(p)


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def prepare():
    from review_fair_comparison_ids import expected_model_spec
    if (OUT / "freeze.json").exists():
        return check_freeze()
    cases = [semantic_case(c) for c in read(DESIGN / "controlled_structural_cases.json")]
    validation = validate_semantics(cases)
    jobs = [{"key": c["key"], "arm": arm, "split": c["split"]} for c in cases for arm in ("D", "U")]
    jobs.sort(key=lambda x: digest([20260915, x]))
    specs = {m: expected_model_spec(m) for m in MODELS}
    assert specs["gpt55_responses"]["api"] == "responses" and "airouting" in specs["gpt55_responses"]["base_url"]
    f = {"version": "A-executable-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
         "source_hashes": hashes(), "case_hash": digest(cases), "job_hash": digest(jobs),
         "system_prompt": SYSTEM, "model_specs": specs, "max_tokens": 1800,
         "transport_attempts": 3, "format_retries": 0, "workers": 8,
         "pilot_positions": 162, "formal_positions_per_model": 648,
         "pilot_gate": "each sharing: no-omission D tasks >=8/9, workflows >=2/3; JSON >=95%; all terminal, no unresolved transport",
         "bootstrap_seed": 20260915, "bootstrap_draws": 10000,
         "execution": "plan all three receivers, then submit local transactions in topological order",
         "cache": "exact system/payload/model/phase reuse; not independent sampling"}
    write(OUT / "cases.json", cases)
    write(OUT / "jobs.json", jobs)
    write(OUT / "offline_validation.json", validation)
    write(OUT / "freeze.json", f)
    return f


def check_freeze():
    f = read(OUT / "freeze.json")
    assert f["source_hashes"] == hashes(), "frozen_source_changed"
    assert f["case_hash"] == digest(read(OUT / "cases.json")), "frozen_case_changed"
    assert f["job_hash"] == digest(read(OUT / "jobs.json")), "frozen_jobs_changed"
    assert f["system_prompt"] == SYSTEM
    return f


class Requests:
    def __init__(self, phase, model, freeze):
        from sandbox_review_deployment import RawCalls
        from review_fair_comparison_ids import expected_model_spec
        assert expected_model_spec(model) == freeze["model_specs"][model]
        self.calls = RawCalls(OUT / phase / model, model)
        assert self.calls.client_holder.clients["gpt"].max_retries == 0
        self.freeze = freeze
        self.locks = {}
        self.guard = threading.Lock()
        self.stop = threading.Event()
        self.failures = 0

    def call(self, data):
        key = digest({"system": SYSTEM, "payload": data})
        with self.guard:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            if self.stop.is_set():
                return {"request_identity": key, "status": "not_attempted_batch_stopped", "content": None, "trace": []}
            trace = []
            for attempt in range(3):
                wire, cache = self.calls.call(SYSTEM, data, f"workflow-A:{key}:attempt{attempt + 1}", max_tokens=self.freeze["max_tokens"])
                trace.append({"request_hash": wire["request_hash"], "status": wire["status"], "cache_hit": cache})
                if wire["status"] != "transport_failure":
                    with self.guard:
                        self.failures = 0
                    break
                if wire.get("http_status") in (401, 403):
                    self.stop.set()
                    break
                if attempt < 2:
                    time.sleep(attempt + 1)
            if wire["status"] == "transport_failure":
                with self.guard:
                    self.failures += 1
                    if self.failures >= 3:
                        self.stop.set()
            return {"request_identity": key, "status": wire["status"], "content": wire["content"], "trace": trace}


def observable_node(case, i, arm, outcome):
    p, ids = payload(case, i, arm)
    r = case["receivers"][i]
    sent = set(ids)
    allowed = set(r["allow_ids"])
    parsed = parse_response(outcome["content"]) if outcome["status"] == "ok" else None
    outside = {e["id"]: e["value"] for e in case["records"] if e["id"] not in allowed}
    found = sorted(e for e, v in outside.items() if v in (outcome["content"] or ""))
    confirmed = outcome["status"] in ("ok", "incomplete_response")
    return {"receiver": i, "payload_hash": digest(p), "sent_ids": ids,
            "input_confirmed": confirmed, "input_forbidden_ids": sorted(sent - allowed),
            "required_arrived": len(sent & set(r["need_ids"])),
            "complete_required_input": set(r["need_ids"]) <= sent,
            "output_forbidden_ids": found, "parsed": parsed, **outcome}


def run_one(requests, phase, model, case, arm, freeze):
    path = OUT / phase / model / "rows" / f"{case['key']}_{arm}.json"
    if path.exists():
        row = read(path)
        assert row["freeze_hash"] == digest(freeze)
        if all(n["status"] != "not_attempted_batch_stopped" for n in row["nodes"]):
            return row
    nodes = [observable_node(case, i, arm, requests.call(payload(case, i, arm)[0])) for i in range(3)]
    scored = score_workflow(case, [n["parsed"] for n in nodes])
    for i, n in enumerate(nodes):
        committed = scored["nodes"][i]["execution"]["status"] == "committed"
        args = json.dumps(n["parsed"]["args"], ensure_ascii=False) if committed else ""
        n["committed_tool_forbidden_ids"] = sorted(e["id"] for e in case["records"]
            if e["id"] not in case["receivers"][i]["allow_ids"] and e["value"] in args)
    row = {"case_key": case["key"], "background": case["background"], "family": case["family"],
           "mask_phase": case["mask_phase"], "sharing": case["sharing"], "pattern": case["pattern"],
           "phase": phase, "model": model, "arm": arm, "freeze_hash": digest(freeze),
           "nodes": nodes, "score": scored}
    write(path, row)
    return row


def run(phase, model):
    f = check_freeze()
    if phase == "pilot":
        assert model == "glm"
    else:
        pilot = analyze("pilot", "glm", audit=False)
        assert pilot["pilot_gate"]["passed"], "pilot_gate_not_passed"
    cases = {c["key"]: c for c in read(OUT / "cases.json")}
    jobs = [j for j in read(OUT / "jobs.json") if j["split"] == phase]
    requests = Requests(phase, model, f)
    completed = 0
    with ThreadPoolExecutor(max_workers=f["workers"]) as pool:
        futures = [pool.submit(run_one, requests, phase, model, cases[j["key"]], j["arm"], f) for j in jobs]
        for future in as_completed(futures):
            row = future.result()
            completed += 1
            print(json.dumps({"phase": phase, "model": model, "completed_workflow_arms": completed,
                "total": len(jobs), "last_success": row["score"]["workflow_success"],
                "batch_stopped": requests.stop.is_set()}), flush=True)
    analyze(phase, model)


def aggregate(rows):
    ns = [n for r in rows for n in r["nodes"]]
    return {"workflows": len(rows), "workflow_success": sum(r["score"]["workflow_success"] for r in rows),
            "tasks": len(ns), "task_success": sum(r["score"]["goal_tasks"] for r in rows),
            "argument_exact": sum(n["argument_exact"] for r in rows for n in r["score"]["nodes"]),
            "complete_required_input": sum(n["complete_required_input"] for n in ns),
            "required_arrived": sum(n["required_arrived"] for n in ns),
            "input_forbidden_planned": sum(len(n["input_forbidden_ids"]) for n in ns),
            "input_forbidden_confirmed": sum(len(n["input_forbidden_ids"]) for n in ns if n["input_confirmed"]),
            "output_forbidden": sum(len(n["output_forbidden_ids"]) for n in ns),
            "committed_tool_forbidden": sum(len(n["committed_tool_forbidden_ids"]) for n in ns),
            "blocked_nodes": sum(r["score"]["blocked_nodes"] for r in rows),
            "format_valid": sum(n["parsed"] is not None for n in ns),
            "responses_received": sum(n["input_confirmed"] for n in ns),
            "transport_unresolved": sum(n["status"] == "transport_failure" for n in ns),
            "not_attempted": sum(n["status"] == "not_attempted_batch_stopped" for n in ns),
            "partial_state_errors": sum(not r["score"]["partial_commits_state_exact"] for r in rows)}


def bootstrap(rows):
    import numpy as np
    backgrounds = sorted({r["background"] for r in rows})
    rng = np.random.default_rng(20260915)
    sample = rng.integers(0, len(backgrounds), (10000, len(backgrounds)))
    indexed = {(r["background"], r["sharing"], r["pattern"], r["arm"]): r for r in rows}
    arrays, result = {}, {}
    def interval(a):
        return {"mean": float(a.mean()), "ci95": np.percentile(a[sample].mean(axis=1), [2.5, 97.5]).tolist()}
    for level in LEVELS:
        for pattern in PATTERNS:
            key = f"{level}/{pattern}"
            values = {"workflow_delta": [], "task_rate_delta": [], "forbidden_input_delta_per_workflow": []}
            for bg in backgrounds:
                d = indexed[bg, level, pattern, "D"]
                u = indexed[bg, level, pattern, "U"]
                values["workflow_delta"].append(int(u["score"]["workflow_success"]) - int(d["score"]["workflow_success"]))
                values["task_rate_delta"].append((u["score"]["goal_tasks"] - d["score"]["goal_tasks"]) / 3)
                values["forbidden_input_delta_per_workflow"].append(sum(len(n["input_forbidden_ids"]) for n in u["nodes"]) - sum(len(n["input_forbidden_ids"]) for n in d["nodes"]))
            arrays[key] = {k: np.array(v) for k, v in values.items()}
            result[key] = {k: interval(a) for k, a in arrays[key].items()}
    interaction = (arrays["high/staggered_slots"]["workflow_delta"] - arrays["high/synchronized_slots"]["workflow_delta"]
                   - arrays["low/staggered_slots"]["workflow_delta"] + arrays["low/synchronized_slots"]["workflow_delta"])
    family = {r["background"]: r["family"] for r in rows}
    result["predeclared_workflow_interaction"] = {**interval(interaction), "leave_one_family_out": {
        str(f): float(interaction[[family[b] != f for b in backgrounds]].mean()) for f in (0, 1, 2)}}
    return {"backgrounds": len(backgrounds), "templates": 3, "draws": 10000, "seed": 20260915,
            "scope": "conditional on three synthetic families and fixed mask phases; not population uncertainty", "estimates": result}


def audit_row(row, case, f, base):
    assert row["freeze_hash"] == digest(f)
    for i, n in enumerate(row["nodes"]):
        p, ids = payload(case, i, row["arm"])
        assert digest(p) == n["payload_hash"] and ids == n["sent_ids"]
        if n["trace"]:
            wire = read(base / "wire" / (n["trace"][-1]["request_hash"] + ".json"))
            assert wire["content"] == n["content"] and wire["status"] == n["status"]
            messages = wire["request"].get("input", wire["request"].get("messages"))
            assert messages[0]["content"] == SYSTEM
            assert json.loads(messages[1]["content"]) == p
        expected = observable_node(case, i, row["arm"], {k: n[k] for k in ("request_identity", "status", "content", "trace")})
        assert all(n[k] == v for k, v in expected.items())
    assert score_workflow(case, [n["parsed"] for n in row["nodes"]]) == row["score"]
    assert all(n["state_preserved_on_failure"] for n in row["score"]["nodes"])
    metrics = case["offline_information_metrics"][row["arm"]]
    assert sum(len(n["input_forbidden_ids"]) for n in row["nodes"]) == metrics["forbidden_pairs"]
    assert sum(n["complete_required_input"] for n in row["nodes"]) == metrics["complete_receivers_ceiling"]


def analyze(phase, model, audit=True):
    f = check_freeze()
    base = OUT / phase / model
    rows = [read(p) for p in sorted((base / "rows").glob("*.json"))]
    cases = {c["key"]: c for c in read(OUT / "cases.json")}
    if audit:
        for row in rows:
            audit_row(row, cases[row["case_key"]], f, base)
    count = 54 if phase == "pilot" else 216
    complete = len(rows) == count and not any(n["status"] == "not_attempted_batch_stopped" for r in rows for n in r["nodes"])
    cells = {f"{s}/{p}": {a: aggregate([r for r in rows if r["sharing"] == s and r["pattern"] == p and r["arm"] == a]) for a in ("D", "U")} for s in LEVELS for p in PATTERNS}
    wires = [read(p) for p in (base / "wire").glob("*.json")]
    costs = [w["telemetry"]["provider_reported_cost"] for w in wires]
    overall = aggregate(rows)
    summary = {"phase": phase, "model": model, "freeze_hash": digest(f), "complete": complete,
        "planned_logical_positions": count * 3, "overall": overall, "cells": cells,
        "actual_wire_attempts": len(wires), "wire_transport_failures": sum(w["status"] == "transport_failure" for w in wires),
        "unique_requested_inputs": len({n["request_identity"] for r in rows for n in r["nodes"] if n["trace"]}),
        "logical_cache_reuses": sum(bool(n["trace"]) and all(t["cache_hit"] for t in n["trace"]) for r in rows for n in r["nodes"]),
        "provider_reported_cost_sum": sum(c for c in costs if c is not None), "missing_cost_attempts": sum(c is None for c in costs),
        "actual_input_tokens": sum(w["telemetry"]["input_tokens"] or 0 for w in wires),
        "actual_output_tokens": sum(w["telemetry"]["output_tokens"] or 0 for w in wires),
        "wire_status_counts": {s: sum(w["status"] == s for w in wires) for s in ("ok", "incomplete_response", "transport_failure")},
        "offline_audit_performed": audit}
    if phase == "pilot":
        cal = {s: cells[f"{s}/no_omission"]["D"] for s in LEVELS}
        rate = overall["format_valid"] / overall["responses_received"] if overall["responses_received"] else 0
        passed = (complete and overall["transport_unresolved"] == 0 and rate >= .95
                  and all(x["tasks"] == 9 and x["task_success"] >= 8 and x["workflows"] == 3 and x["workflow_success"] >= 2 for x in cal.values()))
        summary["pilot_gate"] = {"passed": passed, "by_sharing_no_omission_D": cal, "format_rate": rate}
    elif complete:
        summary["paired_statistics"] = bootstrap(rows)
    write(base / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run", "analyze", "check"))
    p.add_argument("--phase", choices=("pilot", "formal"), default="formal")
    p.add_argument("--model", choices=MODELS)
    a = p.parse_args()
    if a.action == "prepare":
        f = prepare()
        print(json.dumps({"freeze_hash": digest(f), "pilot_positions": f["pilot_positions"], "formal_positions": 2 * f["formal_positions_per_model"]}))
    elif a.action == "run":
        assert a.model
        run(a.phase, a.model)
    else:
        f = check_freeze()
        if a.action == "check":
            validate_semantics(read(OUT / "cases.json"))
        for model in ([a.model] if a.model else (["glm"] if a.phase == "pilot" else MODELS)):
            s = analyze(a.phase, model)
            print(json.dumps({k: s[k] for k in ("phase", "model", "complete", "actual_wire_attempts", "overall")}, ensure_ascii=False))
            if "pilot_gate" in s:
                print(json.dumps({"pilot_gate": s["pilot_gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
