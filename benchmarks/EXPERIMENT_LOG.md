# Experiment Log

## Completed

- Neutral SciFact full test evaluation: BM25, exact MiniLM cosine, HNSW, and RRF.
- Neutral deterministic ArguAna 200-query contrast evaluation with identical-ID exclusion.
- Exact neutral A/B quality and raw-ranking reproduction.
- Clean installed-wheel AtlasRAG SciFact A/B evaluation.
- Clean installed-wheel AtlasRAG ArguAna-200 A/B evaluation.
- Exact package A/B identity, quality, and raw-ranking reproduction for all three package methods.
- Neutral-versus-package method-level comparison with retained ranking and metric differences.
- SciFact train-only candidate-k, RRF-k, and HNSW-ef ablations.
- Synthetic authorization, provenance, missing-component, missing-shard, stale-index, and unsupported-query evaluation.
- Integrated AtlasRAG BM25, exact dense, and RRF synthetic API smoke.
- Query-level failure/rescue analysis and local regression gates.

## Edge cases resolved

### ArguAna query IDs absent from corpus

The 200-query slice contains 183 query IDs also present in the corpus and 17 absent. The package adapter excludes a self document only when the external query ID exists in the corpus. This matches the neutral harness and avoids treating a valid absent ID as malformed data.

### Installed versus neutral ordering

Package A/B runs are internally exact. Package-versus-neutral rankings are close but not identical because equal or nearly equal scores flow through different numeric paths and deterministic identity tie breakers. The differences are reported and are not release failures.

## Failed or superseded work

An early monolithic long-query SQLite FTS5 ArguAna experiment was too CPU heavy and exited before a final artifact. No metric from it is retained. It was replaced by a transparent corpus-wide Okapi BM25 path, batched quality evaluation, separate deterministic latency sampling, and a checksum-frozen 200-query contrast slice.

SciFact train-only ablations exposed stronger exploratory cells, but the official test configuration was not changed after test results were viewed.

Two-thread neutral sparse/dense fanout improved one sampled SciFact timing and worsened ArguAna. No universal concurrency-speedup claim is approved.
