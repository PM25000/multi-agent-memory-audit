# Multi-agent memory audit

Code and research evidence for **Auditing Recipient Exposure in Multi-Agent
Memory Distribution: Paired Accounting and Finite-Frame Evidence**.

Authors: Minghui Deng, Guijing Huo and Kai Weng.

This repository provides research artifact version `public-20261018-v2`.
The eight evidence archives and checksum list are [Release attachments](https://github.com/PM25000/multi-agent-memory-audit/releases/tag/public-20261018-v2).
No journal acceptance or DOI is asserted.

`docs/VALIDATION.json` is the pre-publication verification snapshot; its
`uploaded: false` field records the state at verification, not current availability.

The audit holds realized task selections fixed, enumerates the deliveries added
by union broadcast, and estimates their semantic composition using a stratified
human audit. The core frame uses 150 synthetic inputs and three selectors;
96 sampled input-record groups supply 274 judgments per reviewer. A/B are
independent judgments; C is adjudication after seeing A/B alternatives.

## What is provided

| Package | Evidence and offline check |
|---|---|
| S1 | Scored core/business/probe data and numerical replay |
| S2 | Weighted group contributions, simultaneous bounds and sensitivity |
| S3 | Saved selection responses, synthetic inputs, 432 business-arm replays, 384 probe responses |
| S4 | 1728 saved registration response positions and workflow-state checks |
| S5 | 12 external source-audit workflows, anonymous judgments and author adjudication; 11 primary workflows |
| S6 | 360 final native workflow arms and 120 named / 90 unique token-extension executions |
| S7 | Native handoff/SDK evidence, 24 simple-rule configurations, 48 tradeoff pairs and 72 repeat positions |
| S8 | Core context-to-A/B/C-judgment-to-count chain, sampling and 450 S3 record links |

Versioned ZIPs are distributed as Release attachments, separately from Git source history.
`package-inventory.json` and `SHA256SUMS.txt` identify the exact prepared assets.
The source mirror in `code/packages/` is for inspection. Run scripts from their
complete extracted packages, where relative evidence paths are available.

## Verify

Install the needed dependencies first; all verification uses saved records,
not new model requests. S1/S3 need `requirements-core.txt`; S6 analysis and
optional native action/token replay use `requirements-native.txt`.

```text
python -B code/verify_release.py --assets PATH_TO_RELEASE_ASSETS
```

For native/token replay, separately obtain the exact tokenizer files and tiny
manifests listed in S6/TOKENIZER_ASSETS.json, then run:

```text
python -B code/verify_release.py --assets PATH_TO_RELEASE_ASSETS --native --token-assets PATH_TO_PINNED_ASSETS
```

The launcher sets hash seed 0 before child interpreters start. Dependencies must
be installed before disconnecting. No environment installer or fresh model rerun
is claimed. See docs/REPRODUCTION.md for boundaries and per-package commands.

## Scope and licenses

Original code: MIT. Author-created data/documentation: CC BY 4.0. Third-party
exceptions and upstream notices are in THIRD_PARTY_NOTICES.md. Anonymous judgment
and adjudication projections are authorized for release; original answer exports,
identity/consent records, internal discussions and submission documents stay local.

No manuscript/old draft, publisher template, whole virtual environment, tokenizer
binary, prelive dataset or aborted predecessor campaign is included. Failures and
negative results inside the final studies are retained. Third-party native source
is byte-identical to the pinned subset and carries its original MIT notice.

Historical freeze manifests and upstream source indices retain necessary file
paths and hashes for provenance, including references to excluded manuscripts,
prelive revisions, environments and upstream development files. Those file
contents are not distributed. Each package's PUBLIC_MANIFEST.json defines
the current public members. Historical indices do not assert that old
full-tree, manuscript or environment gates passed.

The public packages have new manifests and explicit projected verification gates.
Passing them does not assert that omitted historical manuscript/environment gates
passed. Numerical/provenance replay does not establish human semantic truth,
authorization, natural-deployment prevalence, a general defense, or a new repair
decision method. Native program contracts remain distinct from task-necessity
judgments. The simple-rule feasibility check finds no additional repair decisions.
