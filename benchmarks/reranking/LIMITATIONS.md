# Reranking Limitations

- One dataset, one dense model, and one cross-encoder revision are evaluated.
- SciFact whole documents are benchmark retrieval units; this is not an optimal-chunking claim.
- Every paired quality-delta interval includes zero.
- The bootstrap analysis is post-run analysis, not a preregistered test.
- Run A latency was contaminated by substantial concurrent host activity.
- Run B latency is a single-host observation on an RTX 4060 Laptop GPU and is not a production SLO.
- The exact dense candidate generator is intentionally dependency-light and CPU-bound.
- No cost, power, multi-user concurrency, batching-across-queries, or network latency is measured.
- Candidate recall bounds reranker performance; missing candidates cannot be rescued.
- Authorization safety is regression-tested in the package, but this benchmark uses public dataset records and is not a security audit.
- No third-party dataset records, model weights, or caches are redistributed.
- The experiment does not justify default-enabled reranking.
