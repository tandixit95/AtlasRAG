# Reranking Claim Ledger

| ID | Allowed claim | Evidence | Forbidden expansion |
|---|---|---|---|
| R1 | AtlasRAG `0.3.0.dev0` commit `43c4ef3` implements an authorization-safe reranking boundary, optional pinned cross-encoder adapter, immutable citations, and candidate-stage traces | Source and 73 passing tests | Production deployment, historical recovered code, or external adoption |
| R2 | Clean installed-wheel runs A and B produced byte-identical SciFact quality summaries and raw rankings | `artifacts/reproducibility.json` | Independent or cross-host replication |
| R3 | Depth 10 preserved Recall@10 and Success@10 while increasing MRR@10 by 0.0139 and nDCG@10 by 0.0068 in this 300-query experiment | `artifacts/run-a.json` and paired analysis | Statistically established improvement or universal gain |
| R4 | Depth 20 was dominated by depth 10 on the reported point estimates and latency | `artifacts/reproducibility.json` | Depth 20 is always inferior on other datasets or models |
| R5 | Depth 50 had the highest aggregate point estimates: Recall@10 0.8272, MRR@10 0.6614, nDCG@10 0.6943, Success@10 0.8400 | `artifacts/run-a.json` | Significant improvement, benchmark leadership, or default recommendation |
| R6 | Every paired 95% bootstrap interval for the reported quality deltas included zero | `artifacts/reproducibility.json` | Proof that reranking has no effect |
| R7 | Run B observed reranker-only p50 values of 42.3, 87.2, and 222.3 ms at depths 10, 20, and 50 on one RTX 4060 Laptop GPU | `artifacts/run-b.json` | Stable cross-run latency, production SLO, or other-hardware expectation |
| R8 | Run A showed severe timing dispersion under concurrent host load, so stable latency publication is rejected | raw timing samples and diagnostics | Discarding run A or reporting run B as independently reproduced latency |
| R9 | Reranking remains opt-in and disabled by default | source design and recommendation record | AtlasRAG defaults to cross-encoder reranking |

Machine-readable artifacts outrank prose. Correct the narrative when they disagree.
