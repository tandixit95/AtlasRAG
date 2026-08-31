# Reranker Candidate Final Evaluation v2

This namespace freezes the final evaluation **before outcomes** for the single candidate nominated by the synthetic-only development profiler.

## Frozen candidate

- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Revision: `c5ee24cb16019beea0893ab7796b1df96625c6b8`
- Candidate depth: `10`
- Reranker batch size: `32`
- Final top-k: `10`
- Development profile: `benchmarks/development/reranker_candidate_v1/PROFILE.json`
- Development profile SHA-256: `312072db352d7ef8029a40f74142338e0437f6957888528df47acc3eca13a22b`

The development profiler selected batch 32 mechanically: batch 64 had the best same-depth p95, while batch 32 remained within the precommitted 1.10x tolerance and was therefore the smallest eligible batch size. This is development-only component evidence, not a final quality or production-performance result.

## Frozen final tasks and gates

The final evaluation reuses the established `scifact-test-300` and deterministic `arguana-contrast-200` task shapes, 10,000 paired bootstrap resamples, the existing quality/safety/latency gates, and the 75 ms controlled reranker-component p95 budget. `GATES.json` adds an explicit veto requiring the executed configuration to report `reranker_batch_size == 32`.

The candidate may not be tuned, replaced, or reselected after any final outcome is observed. Hybrid RRF remains the default unless every frozen promotion gate passes.

The protocol is currently `frozen_unexecuted`. No final SciFact or ArguAna outcome is established by this freeze artifact.

Run `python benchmarks/final_evaluation/reranker_candidate_v2/verify_protocol.py` before any final evaluation execution.
