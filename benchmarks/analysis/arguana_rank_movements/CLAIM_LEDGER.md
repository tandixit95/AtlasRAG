# Claim Ledger

| ID | Allowed claim | Required evidence | Forbidden expansion |
|---|---|---|---|
| A1 | The frozen ArguAna depth-10 reranker preserved the exact top-10 candidate set for all 200 contrast-slice queries | Source hash, analyzer, verifier | Every reranker preserves candidate sets |
| A2 | Recall@10 and Success@10 were unchanged on all 200 queries | `artifacts/analysis.json` | Quality was unchanged overall |
| A3 | Relevant ranks regressed on 80 queries, improved on 45, stayed unchanged on 44, and were absent in both lists on 31 | Deterministic artifact and tests | Causal explanation from counts alone |
| A4 | Mean relevant rank delta was +0.579881656805 over 169 queries with a relevant candidate | Deterministic artifact | Production relevance loss or user impact |
| A5 | The evidence supports an ordering-regression diagnosis within the frozen candidate set | A1-A4 plus frozen promotion evidence | Universal failure mode or model defect |
| A6 | The public package contains no query/corpus payload text | Fail-closed key scan and verifier | Formal privacy or licensing certification |

## Rule

This analysis can diagnose the frozen outcome. It cannot justify tuning on the final
slice or promote a revised candidate without a new protocol and independent evidence.
