# Promotion Evaluation Results

## Aggregate quality

| Task | Method | Recall@10 | MRR@10 | nDCG@10 | Success@10 |
|---|---|---:|---:|---:|---:|
| SciFact test 300 | Hybrid RRF | 0.8212 | 0.6449 | 0.6845 | 0.8367 |
| SciFact test 300 | Reranked depth 10 | 0.8212 | 0.6589 | 0.6914 | 0.8367 |
| ArguAna contrast 200 | Hybrid RRF | 0.8450 | 0.4288 | 0.5274 | 0.8450 |
| ArguAna contrast 200 | Reranked depth 10 | 0.8450 | 0.3679 | 0.4797 | 0.8450 |

Runs A and B reproduced every aggregate quality value, the complete query-level ranking files, and the summary CSV files exactly.

## Paired quality deltas

| Task | Metric | Mean delta | Paired 95% bootstrap interval | Gate interpretation |
|---|---|---:|---:|---|
| SciFact | Recall@10 | 0.0000 | [0.0000, 0.0000] | No-regression pass |
| SciFact | MRR@10 | +0.0139 | [-0.0149, +0.0420] | Improvement not established |
| SciFact | nDCG@10 | +0.0068 | [-0.0141, +0.0272] | Mean non-regression pass |
| ArguAna contrast | Recall@10 | 0.0000 | [0.0000, 0.0000] | No-regression pass |
| ArguAna contrast | MRR@10 | -0.0609 | [-0.1067, -0.0142] | Statistically supported regression |
| ArguAna contrast | nDCG@10 | -0.0476 | [-0.0831, -0.0124] | Statistically supported regression |

SciFact repeats the Day 5 point estimate but still does not establish a reliable MRR improvement. The second task shape reverses the direction: reranking significantly worsens both reported ordering metrics on the frozen ArguAna contrast slice.

## Reproduction, safety, and provenance

- SciFact A/B ranking SHA-256: `2b689b58c1eb1af1f85e3ce0de9a6453ad692a7eb88f67b1c701bd45e1d5bc16`
- ArguAna A/B ranking SHA-256: `efb44f2c475ca57a961fa37908958656e33f0091b4442fbd8e1c5bc06237a7f0`
- Authorization leakage: 0
- Unauthorized candidate scoring: 0
- Excluded-chunk leakage: 0
- Malformed protected metadata: failed closed
- Complete citations: 3,000 of 3,000 SciFact results and 2,000 of 2,000 ArguAna results
- Redistributed query or corpus text: none

## Controlled reranker latency

| Task | Run A p50 | Run A p95 | Run B p50 | Run B p95 | Cross-run p95 ratio | 75 ms budget |
|---|---:|---:|---:|---:|---:|---|
| SciFact | 92.7 ms | 131.8 ms | 91.2 ms | 134.9 ms | 1.024 | Fail |
| ArguAna contrast | 99.9 ms | 167.9 ms | 117.2 ms | 167.3 ms | 1.004 | Fail |

All four runs passed the host-control and dispersion checks. Timing reproduced within the frozen ratio boundary, but the stable observations exceeded the preregistered component budget. These are local controlled-host observations, not production SLOs.

## Gate decision

- Checks evaluated: 37
- Failed: 6
- Missing: 0
- Disposition: `retain_default_rejected`
- Default after evaluation: hybrid RRF
- Reranking default enabled: no

Failed gates:

1. SciFact reranker p95 budget.
2. SciFact primary MRR interval lower bound above zero.
3. ArguAna reranker p95 budget.
4. ArguAna positive primary MRR mean.
5. ArguAna primary MRR interval lower bound above zero.
6. ArguAna secondary nDCG non-regression.

The latency failures are veto violations, so the candidate is rejected rather than merely inconclusive. The ArguAna quality regression independently strengthens the no-promotion decision.
