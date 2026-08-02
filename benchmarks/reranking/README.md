# AtlasRAG Reranking Development Evidence

This directory is a development evidence track for AtlasRAG `0.3.0.dev0`. It is separate from the immutable `v0.2.0` benchmark release under the parent directory.

The experiment evaluates an optional cross-encoder reranker over authorization-safe hybrid candidates. It is designed to answer one decision question:

> Do the measured ranking gains justify the additional latency and complexity strongly enough to enable reranking by default?

The current answer is **no**.

## Evaluated package

- Git commit: `43c4ef33b212869c94ff8cd9bb1c8615b0084b24`
- Package version: `0.3.0.dev0`
- Wheel SHA-256: `43b09b21f813f99f4b8c78d43a358c18a667dbc477da06b4ad92b3a312f8c928`
- Import origin: clean installed wheel in `site-packages`
- Source tests before the experiment: 73 passed

## Experiment

- Dataset: SciFact test split, 300 judged queries, 5,183 corpus documents
- Candidate generator: AtlasRAG BM25 + exact dense retrieval + RRF
- Dense model: `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2` at revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`
- Reranked candidate depths: 10, 20, and 50
- Final cutoff: 10
- A/B quality and raw rankings: byte-identical

## Decision

- Reranking remains opt-in and is not enabled by default.
- Depth 10 is the most efficient point-estimate configuration.
- Depth 20 is rejected because depth 10 has equal or better values for every reported quality metric and lower latency.
- Depth 50 has the strongest aggregate point estimates, but every paired 95% bootstrap interval includes zero.
- Timing is not accepted as a stable cross-run claim because run A experienced severe host contention while run B was comparatively stable.

See [`RESULTS.md`](RESULTS.md), [`METHODOLOGY.md`](METHODOLOGY.md), [`LIMITATIONS.md`](LIMITATIONS.md), and [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md).
