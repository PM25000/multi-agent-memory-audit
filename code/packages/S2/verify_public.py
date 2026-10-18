"""Public integrity and read-only replay entry for S1-S5/S8."""
import json
import runpy
import sys
from public_checks import ROOT, integrity, activate_guard

manifest = integrity()
activate_guard()
package = manifest["package"]
if package == "S5":
    from public_checks import read
    a, b = read(ROOT / "judgments/A.json"), read(ROOT / "judgments/B.json")
    comparison = read(ROOT / "human-comparison.json")
    resolution = read(ROOT / "author-resolution.json")
    for row in comparison["rows"]:
        assert row["A"] == a["answers"][row["id"]]
        assert row["B"] == b["answers"][row["id"]]
    for row in resolution["author_decisions"]:
        assert row["A"] == a["answers"][row["id"]]
        assert row["B"] == b["answers"][row["id"]]
        assert row["final"] == resolution["questions"][row["id"]]["answer"]
    roles = 0
    for workflow in read(ROOT / "dataset.json")["workflows"]:
        assert workflow.get("split") != "pilot"
        for role in workflow["receivers"]:
            provenance = role.get("reference_provenance", {})
            if provenance.get("human_checked"):
                q = resolution["questions"][provenance["question_id"]]
                assert set(role["required_ids"]) == set(q["answer"]["ids"])
                assert q["origin"] == provenance["origin"] and q["human_checked"]
                roles += 1
    assert roles == 8
    print(json.dumps({"S5_anonymous_judgment_chain_checked": True, "human_checked_roles": roles}))
scripts = {
    "S1": ("verify.py", []), "S2": ("verify.py", []),
    "S3": ("experiments/replay_information_raw.py",
           ["--check-manifest", "paper-information/raw-replay-manifest-20260927.json"]),
    "S4": ("verify.py", []), "S5": ("replay.py", []), "S8": ("verify-reading-guide.py", [])
}
script, arguments = scripts[package]
sys.path.insert(0, str((ROOT / script).parent))
sys.argv = [script] + arguments
runpy.run_path(str(ROOT / script), run_name="__main__")
print(json.dumps({"status": "PASS", "public_package": package,
                  "public_files_checked": len(manifest["files"]), "new_model_calls": 0}))
