# Limitations

- The ArguAna result is a deterministic 200-query contrast slice, not the full
  official test score.
- The upstream license records conflict, so this package redistributes no query or
  corpus payload text.
- Query/document identifiers can reveal task grouping but not the underlying payload.
- The analysis is descriptive. It cannot determine which linguistic, domain, or model
  behavior caused the reranker scores.
- The source candidate depth is fixed at 10, so the analysis does not evaluate larger
  candidate pools.
- The frozen final slice must not be used to tune a replacement candidate.
- A revised reranker requires a separate development task and a new freeze-before-
  outcomes protocol before final evaluation.
- These results do not generalize to every reranker, dataset, or production workload.
