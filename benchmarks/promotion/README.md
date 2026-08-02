# Default-Path Promotion Evidence

This directory applies the normative rules in `../../EVALUATION_STANDARD.md` to one candidate: cross-encoder reranking at candidate depth 10 versus the current hybrid RRF default.

The machine-readable protocol in `PROMOTION_GATES.json` was frozen at commit `c59485d698c41797dc307b81fa8a4198f1113812` before the ArguAna reranking outcomes were generated. Its SHA-256 is `ff186c5cd42839478d7b3e7f40377383cec2d7c472b5194eeeb4e70484c217c4`.

## Decision

**Retain the current hybrid RRF default. Depth-10 reranking is rejected for default promotion under this protocol.**

The evidence is complete: 37 gates were evaluated, 6 failed, and none were missing. Two veto gates failed because controlled reranker p95 exceeded the frozen 75 ms component budget on both task shapes. The ArguAna contrast slice also showed a statistically supported ranking regression. SciFact retained its prior positive point estimates, but the paired MRR interval still included zero.

Reranking remains opt-in and disabled by default. This directory is development evidence, not a `v0.3.0` release.

## Tasks

- SciFact test: 300 judged queries and 5,183 corpus documents.
- ArguAna contrast: the existing deterministic 200-query slice and 8,674 corpus documents, with identical query-document IDs excluded before scoring.

No query text, corpus text, dataset payload, model weight, or embedding cache is redistributed.

## Evidence map

- `PROMOTION_GATES.json`: frozen machine-readable gates.
- `METHODOLOGY.md`: package, task, statistical, and host-control method.
- `RESULTS.md`: quality, paired intervals, latency, and gate outcome.
- `LIMITATIONS.md`: claim boundary.
- `CLAIM_LEDGER.md`: approved and forbidden wording.
- `MANIFEST.json`: package, model, dataset, source, and artifact identities.
- `artifacts/promotion-report.json`: normalized evidence and all gate results.
- `artifacts/*-run-{a,b}.json`: independently executed task summaries.
- `artifacts/*-rankings.jsonl.gz`: canonical query-level rankings without text.
- `verify_artifacts.py`: privacy, identity, checksum, reproduction, and decision verifier.
