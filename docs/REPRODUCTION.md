# Reproduction guide

1. Verify each asset using SHA256SUMS.txt. Extract S1-S8 into separate directories.
2. Run `python -B -X utf8 verify_public.py` in each package. These entry points
   check public inventories first and block network/process access and evidence
   writes. Optional dependency caches are restricted to .runtime-cache.
3. S6 default reaggregates both final analyses. For action/request/token checks,
   use the recorded Python dependencies, obtain the separately pinned tokenizer
   assets, set PYTHONHASHSEED=0 before interpreter startup, and run
   `python -B -X utf8 verify_public.py --native-replay --token-assets PATH`.
4. S7 default rechecks message IDs/bodies/topics/senders, actual SDK requests,
   response identity/costs, all 24 feasibility configurations, 48 tradeoff pairs,
   and the task-specific simple-rule countercheck. For actual sandbox action/state
   replay, with hash seed 0 and the native dependencies, run
   `python -B -X utf8 verify_public.py --native-replay --agentdojo-root PATH_TO_S6`.
   This replays 192 native role positions and 72 repeat positions. It does not
   reexecute the AutoGen message-passing runtime or claim new provider results.
5. The repository launcher additionally checks all 450 original S3/S8 record
   links using original record-byte identities, rather than the old ZIP identity.

Keep all supplied relative paths. Run only the public verification entry points.
Historical controller/generator sources retained for method inspection may contain
live, intake or obsolete whole-checkout entry points; they are not requested here.
No new human review is required or requested. The two completed core human audits
retain their original scope and provenance.

## Public derivation

Original frozen ZIPs remain in the research archive. Public copies have distinct
names/hashes. PUBLIC_MANIFEST.json records retained original hashes and transformed
files. S5 removes two pilot fixtures and replaces raw exports with anonymous
judgment projections. S6 removes the aborted earlier campaign, duplicate frozen
source copies, old analyses/projections and unrelated upstream development files.
S7 removes old manuscripts/templates, prelive revisions and bundled virtual-env
code. Temporary locks and old build reports are omitted. Current-study failures,
identity/transport outcomes and negative results are retained.

Original full-tree/manuscript/interpreter gates are not represented as passed.
The derived public gate checks current package identities, retained dependency
origins and all retained calculation/action assertions. Source/data bytes are
unchanged except documented field projections, read-only helper extraction
and normalization of one historical instruction quotation into a method note.
The original hash seed was not recorded; seed 0 is an explicitly selected replay
configuration, not a recovered execution fact.

These artifacts support offline audit/replay, not a clean-install or API-output
regeneration guarantee. They do not automatically validate human label semantics.

## Historical indices and current public members

Historical freeze manifests and upstream source indices retain necessary file
paths and hashes for provenance, including references to excluded manuscripts,
prelive revisions, environments and upstream development files. Those file
contents are not distributed. Each package's PUBLIC_MANIFEST.json defines
the current public members. Historical indices do not assert that old
full-tree, manuscript or environment gates passed.
