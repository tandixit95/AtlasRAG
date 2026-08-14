# ArguAna Rank-Movement Analysis

This package explains the frozen depth-10 reranker regression using only the public
promotion evidence surface: query/document IDs, candidate ranks, reranker scores,
metric deltas, hashes, and judged relevance. It contains no ArguAna query text,
corpus text, or redistributed dataset payload.

## Question

The promotion experiment preserved Recall@10 but reduced MRR@10 and nDCG@10 on the
deterministic 200-query ArguAna contrast slice. This analysis asks whether the loss
came from candidate-set changes or from reordering the same candidates.

## Result

- The baseline and reranked candidate sets were identical for all 200 queries.
- Recall@10 and Success@10 were unchanged for every query.
- Relevant documents moved down on 80 queries, up on 45, stayed at the same rank on
  44, and were absent from both top-10 lists on 31.
- Among the 169 queries with a relevant candidate, the mean rank delta was +0.5799
  positions, where positive means worse.
- All 52 queries with the relevant document at baseline rank 1 either stayed at rank
  1 (24) or regressed (28); none could improve.

This supports an ordering-regression diagnosis for the frozen candidate, not a causal
linguistic explanation and not a revised-model recommendation.

## Reproduce

```bash
python3 -m benchmarks.src.analyze_promotion_rank_movements \
  --rankings benchmarks/promotion/artifacts/arguana-rankings.jsonl.gz \
  --output benchmarks/analysis/arguana_rank_movements/artifacts/analysis.json

python3 benchmarks/analysis/arguana_rank_movements/verify_artifacts.py
```

See `METHODOLOGY.md`, `RESULTS.md`, `LIMITATIONS.md`, and `CLAIM_LEDGER.md`.
