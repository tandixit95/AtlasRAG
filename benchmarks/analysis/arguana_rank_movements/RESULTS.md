# Results

## Aggregate movement

| Class | Queries |
|---|---:|
| Regressed | 80 |
| Improved | 45 |
| Unchanged | 44 |
| Both absent | 31 |
| Rescued | 0 |
| Lost | 0 |

The candidate set was preserved for all 200 queries. Recall@10 and Success@10 were
unchanged on all 200. The observed loss is therefore ordering loss within the frozen
candidate set rather than candidate recall loss.

## Metric movement

| Metric | Mean delta | Improved | Degraded | Unchanged |
|---|---:|---:|---:|---:|
| MRR@10 | -0.060934523810 | 45 | 80 | 75 |
| nDCG@10 | -0.047640532880 | 45 | 80 | 75 |
| Recall@10 | 0.000000000000 | 0 | 0 | 200 |
| Success@10 | 0.000000000000 | 0 | 0 | 200 |

These means reproduce the frozen promotion evidence.

## Rank behavior

Across the 169 queries where the relevant document was in the top-10 candidate set:

- mean rank delta: `+0.579881656805`;
- median rank delta: `0`;
- best movement: `-9` positions;
- worst movement: `+8` positions.

Baseline rank 1 is structurally asymmetric: it cannot improve. Of 52 such queries, 24
stayed at rank 1 and 28 moved down. Baseline ranks 6-10 had negative mean deltas, but
those gains did not offset the loss from promoting competing candidates above highly
ranked relevant documents.

## Descriptive score separation

| Class | Mean relevant score | Median margin from top score |
|---|---:|---:|
| Improved | -0.383363339388 | 0.706820368767 |
| Unchanged | 0.043138332665 | 0.000000000000 |
| Regressed | -1.067266186653 | 2.124304255471 |

The larger score margin on regressed queries is consistent with the observed ranking
movement, but it is not a causal explanation and must not be used to tune on this
frozen final slice.
