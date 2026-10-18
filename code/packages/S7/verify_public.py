"""Recompute public S7 wire, decision-sufficiency and repeat evidence."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
from public_checks import ROOT, integrity, activate_guard

p = argparse.ArgumentParser()
p.add_argument("--native-replay", action="store_true")
p.add_argument("--agentdojo-root", type=Path,
               help="Extracted S6 public package, containing pinned AgentDojo source and runtime helpers")
args = p.parse_args()
integrity()
activate_guard([args.agentdojo_root] if args.agentdojo_root is not None else [])
sys.path.insert(0, str(ROOT / "experiments"))
for script in ("verify_native_task_wire_evidence_20261006.py", "verify_repair_tradeoff_wire_20261006.py"):
    saved = sys.argv
    try:
        sys.argv = [script]
        runpy.run_path(str(ROOT / "experiments" / script), run_name="__main__")
    finally:
        sys.argv = saved
import check_repair_decision_feasibility_20261006 as feasibility
result = feasibility.run()
assert result["all_retrospective"]["configurations"] == 24
assert result["all_retrospective"]["changed_decisions_vs_simple_origin_rule"] == 0
import screen_repair_tradeoffs_20261006 as screen
assert screen.compute() == screen.read(screen.DEST / "results.json")
runpy.run_path(str(ROOT / "experiments/countercheck_repair_tradeoff_rules_20261006.py"), run_name="__main__")

native = None
if args.native_replay:
    assert os.environ.get("PYTHONHASHSEED") == "0"
    assert args.agentdojo_root and args.agentdojo_root.is_dir()
    sys.path.insert(0, str(args.agentdojo_root.resolve() / "experiments"))
    import public_native_actions as replay
    native = replay.verify(ROOT)
print(json.dumps({"status": "PASS", "public_package": "S7",
                  "retrospective_configurations_recomputed": 24,
                  "tradeoff_role_pairs_recomputed": 48,
                  "new_decisions_over_simple_rules": 0, "new_model_calls": 0,
                  "native_action_replay": native,
                  "historical_manuscripts_and_venv_included": False,
                  "original_full_freeze_gate_pass_claimed": False}))
