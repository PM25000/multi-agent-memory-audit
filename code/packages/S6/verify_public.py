"""Derived S6 replay: current study evidence, no old campaigns or manuscripts."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import sys

from public_checks import ROOT, read, sha, integrity, original_subset, activate_guard

p = argparse.ArgumentParser()
p.add_argument("--native-replay", action="store_true")
p.add_argument("--token-assets", type=Path,
               help="Directory containing glm/ and qwen/ pinned tokenizer files; see TOKENIZER_ASSETS.json")
args = p.parse_args()
if args.native_replay:
    assert os.environ.get("PYTHONHASHSEED") == "0", "Start Python with PYTHONHASHSEED=0"
    assert args.token_assets and args.token_assets.is_dir(), "Pinned tokenizer assets are required for native replay"
manifest = integrity()
activate_guard([args.token_assets] if args.token_assets is not None else [])
sys.path.insert(0, str(ROOT / "experiments"))
import run_external_execution_controls as external_study
import run_execution_extension_v2 as extension_study
from same_input_native_api import digest

gates = []


def public_frozen(study):
    data = read(study.OUT / "dataset.json")
    freeze = read(study.OUT / "freeze.json")
    assert digest(data) == freeze["dataset_sha256"]
    gate = original_subset(freeze["code_sha256"])
    gate["study"] = study.OUT.name
    # Dependencies are reported rather than represented as the original absolute
    # interpreter/source-tree provenance. Native replay still uses exact pinned
    # AgentDojo file identities and all original action/request/state assertions.
    gate["runtime_package_versions"] = {
        k: importlib.metadata.version(k) for k in freeze["dependencies"]["package_versions"]
    }
    gate["matches_recorded_package_versions"] = gate["runtime_package_versions"] == freeze["dependencies"]["package_versions"]
    if args.native_replay:
        assert gate["matches_recorded_package_versions"], "Pinned native package versions differ"
    if not any(g["study"] == gate["study"] for g in gates):
        gates.append(gate)
    if hasattr(study, "configure"):
        study.configure()
    return data, freeze


external_study.frozen = lambda: public_frozen(external_study)
extension_study.frozen = lambda: public_frozen(extension_study)
native = None
if args.native_replay:
    import extra_token_match
    import verify_external_execution_controls_v4 as external
    import verify_execution_extension_v2 as extension
    assets = args.token_assets.resolve()
    index = read(ROOT / "TOKENIZER_ASSETS.json")
    for model, info in index.items():
        for row in info["files"]:
            assert sha(assets / model / row["file"]) == row["sha256"]
        assert sha(assets / model / "manifest.json") == info["manifest_sha256"]
    extra_token_match.ASSETS = assets
    external.verify(False)
    native = extension.verify(False, True)

import analyze_external_execution_controls_v3 as first
import analyze_execution_extension_v2 as second
actual_first = first.build()
actual_second = second.analyze(False)
assert actual_first == read(external_study.OUT / "analysis_v3.json"), "External analysis differs"
assert actual_second == read(extension_study.OUT / "analysis_v2.json"), "Extension analysis differs"
print(json.dumps({
    "status": "PASS", "public_package": "S6", "analysis_comparisons": 2,
    "external_workflow_arms": actual_first["observed_workflow_arms"],
    "native_action_request_replay_performed": args.native_replay,
    "extension_native_replay": None if native is None else {
        k: native[k] for k in ("verified_completed_arms", "verified_wire_receipts", "actual_native_tool_replays")},
    "projected_dependency_gates": gates,
    "paper_projection_and_full_old_checkout_gates_replayed": False,
    "new_model_calls": 0, "aborted_previous_campaign_included": False
}, ensure_ascii=False))
