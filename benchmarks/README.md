# AtlasRAG Benchmark and Evaluation Evidence Package

This directory is the published evidence package for AtlasRAG 0.2.0. It contains a neutral retrieval research harness, clean installed-wheel package evaluations, synthetic authorization and reliability checks, raw rankings, checksums, and claim boundaries.

## Evidence tracks

- Neutral harness: BM25, exact MiniLM cosine search, HNSW, and RRF on frozen public datasets.
- Installed AtlasRAG wheel: package BM25, exact dense, and RRF on the same frozen evaluation identities.
- Synthetic contract evidence: authorization, provenance, determinism, partial/stale/unsupported signaling, and BM25 hidden-document invariance.

## Installed package identity

- Version: 0.2.0
- Commit: `5e86c78a4c40bc6d552d14d4fdcc370b0db8ece1`
- Wheel SHA-256: `30cbaf0030fe86177b7962e43267b6d182534c023eb9d61e7eec7481df048200`
- AtlasRAG source tests: 63 passed
- Package benchmark A/B: exact quality and raw-ranking reproduction on SciFact and the ArguAna contrast slice

## Headline package evidence

- SciFact package hybrid: Recall@10 0.8212, MRR@10 0.6449, nDCG@10 0.6845, Success@10 0.8367.
- ArguAna-200 package hybrid: Recall@10 0.8450, MRR@10 0.4288, nDCG@10 0.5274, Success@10 0.8450.
- Package A/B rankings are identical for BM25, exact dense, and hybrid on every evaluated query.
- HNSW is not an AtlasRAG 0.2.0 capability. HNSW results remain neutral-harness evidence only.

## Key files

- `EXPERIMENT_MANIFEST.json`: run inventory, package identity, hashes, model, and environment.
- `DATASET_PROVENANCE.json`: source-specific rights, checksums, and no-redistribution boundary.
- `METHODOLOGY.md`: evaluation units, algorithms, exclusion semantics, metrics, and timing.
- `RESULTS.md`: neutral and installed-package tables plus explicit comparisons.
- `LIMITATIONS.md`: non-claims and known measurement limits.
- `CLAIM_LEDGER.md`: approved wording and forbidden expansions.
- `ATLASRAG_ADAPTER_SPEC.md`: completed package adapter contract and acceptance evidence.
- `RUNBOOK.md`: clean reproduction commands.
- `artifacts/installed-package-summary.json`: compact machine-readable package result surface.

## Release boundary

This evidence package is published with the AtlasRAG v0.2.0 GitHub release. It redistributes no third-party dataset records, model weights, or caches. It is not a peer-reviewed paper, production SLO, security certification, or claim of external adoption.
