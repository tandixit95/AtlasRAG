# Reranker Candidate Development Protocol v1

This development-only protocol isolates reranker component profiling and candidate selection from AtlasRAG's frozen final evaluation sets. It must not inspect, score, tune on, or derive parameters from the frozen SciFact or ArguAna final evaluation payloads or their query-level outcomes.

The development input is a deterministic synthetic query/document workload. Its purpose is limited to component cost, batching behavior, score-shape sanity, and repeated-order determinism. It cannot establish retrieval quality on natural workloads.

The pinned reranker is `cross-encoder/ms-marco-MiniLM-L6-v2` at revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`. The profiling matrix is candidate depths 10/20/50, batch sizes 16/32/64, five warmups, and thirty measured iterations per configuration. Model loading must use the local snapshot offline under an exclusive benchmark lock.

A candidate may be nominated only if score count and finiteness checks pass, repeated identical inputs preserve ordering, host controls pass, and the chosen batch size remains within the protocol's same-depth p95 tolerance. At most one candidate may be nominated. The default candidate depth remains 10 unless separate development evidence justifies another depth.

This protocol is not a benchmark result, release artifact, production latency claim, throughput claim, quality claim, or default-path promotion package. After candidate selection, a new evaluation namespace must freeze the candidate and gates before any final evaluation data is examined.

## Next implementation boundary

Implement a profiler that consumes `PROTOCOL.json`, generates only the declared synthetic inputs, loads the pinned local snapshot offline, records raw reranker-only timings for the full matrix, and fails closed on protocol or determinism violations. Then run it on an uncontended host and nominate at most one candidate for a separately frozen evaluation.
