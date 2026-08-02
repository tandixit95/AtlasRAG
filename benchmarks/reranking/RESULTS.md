# Reranking Results

## Aggregate quality

| Method | Candidate recall | Recall@10 | MRR@10 | nDCG@10 | Success@10 |
|---|---:|---:|---:|---:|---:|
| Hybrid RRF baseline | 0.9370 at depth 50 | 0.8212 | 0.6449 | 0.6845 | 0.8367 |
| Reranked depth 10 | 0.8212 | 0.8212 | 0.6589 | 0.6914 | 0.8367 |
| Reranked depth 20 | 0.8686 | 0.8144 | 0.6579 | 0.6881 | 0.8300 |
| Reranked depth 50 | 0.9370 | 0.8272 | 0.6614 | 0.6943 | 0.8400 |

Runs A and B reproduced every value above and the complete raw ranking artifact exactly.

## Paired deltas versus hybrid

| Depth | Delta Recall@10 | Delta MRR@10 | Delta nDCG@10 | Delta Success@10 | Rescues | Losses |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.0000 | +0.0139 | +0.0068 | 0.0000 | 0 | 0 |
| 20 | -0.0068 | +0.0130 | +0.0036 | -0.0067 | 4 | 6 |
| 50 | +0.0060 | +0.0165 | +0.0098 | +0.0033 | 14 | 13 |

Every paired 95% bootstrap interval includes zero. The point estimates are useful for choosing the next experiment, but they do not establish a reliable population-level improvement.

Depth 20 is strictly dominated by depth 10 in this experiment: depth 10 has equal or better Recall@10, MRR@10, nDCG@10, and Success@10 while scoring fewer candidates.

## Timing observations

Run A experienced severe host contention and fails the post-hoc dispersion diagnostic. Run B was stable under the same heuristic. The values below are **run-B observations only**, not cross-run latency claims.

| Candidate depth | Reranker-only p50 | Reranker-only p95 | End-to-end p50 | End-to-end p95 |
|---:|---:|---:|---:|---:|
| 10 | 42.3 ms | 50.0 ms | 266.4 ms | 357.2 ms |
| 20 | 87.2 ms | 95.9 ms | 312.5 ms | 349.2 ms |
| 50 | 222.3 ms | 245.3 ms | 451.1 ms | 583.0 ms |

The package exact-dense candidate generator is a dependency-light correctness reference and remains CPU-bound. This experiment does not establish a production serving SLO.

## Decision

Reranking remains disabled by default.

The next evaluation should:

1. repeat the comparison on a second task shape;
2. run latency on an isolated or controlled host;
3. retain depth 10 as the efficiency candidate;
4. reject depth 20 unless a new dataset reverses its dominated result;
5. treat depth 50 as a quality-oriented option only if the larger latency budget is acceptable.
