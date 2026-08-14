# Methodology

## Frozen input

The sole ranking input is:

`benchmarks/promotion/artifacts/arguana-rankings.jsonl.gz`

Expected SHA-256:

`32dab7b69e57cf98d9fc96dd09016cb26d2b6bcce17fff50fa1fb192ac143bc5`

The source contains 200 query-level records for the deterministic ArguAna contrast
slice. Each record includes IDs, baseline top-10 order, reranked top-10 objects,
reranker scores, relevance IDs, metric values, citation hashes, and provenance. It
contains no query or corpus payload text.

## Validation

The analyzer fails closed when:

- a row is missing required keys;
- a ranking does not contain exactly 10 unique IDs;
- reranking changes the candidate set;
- reranker scores or metric evidence are not finite numeric values;
- a nested object contains `text`, `query_text`, `corpus_text`, or `document_text`.

## Rank classification

For each query, the first judged relevant document receives a baseline rank and a
reranked rank. The query is classified as:

- `improved`: candidate rank is numerically smaller;
- `unchanged`: ranks match;
- `regressed`: candidate rank is numerically larger;
- `rescued`: absent from baseline top 10 but present after reranking;
- `lost`: present before reranking but absent after;
- `both_absent`: absent from both lists.

Rank delta is `candidate_rank - baseline_rank`, so positive values are regressions.

## Score diagnostics

For improved, unchanged, and regressed queries, the analysis reports descriptive
summaries for:

- the relevant document's reranker score;
- the difference between the top reranker score and the relevant score.

These values are descriptive. They do not prove why the model produced the ordering.

## Reproducibility

The artifact is regenerated from the frozen gzip input with deterministic sorting,
fixed floating-point rounding to 12 decimal places, and no timestamps. The verifier
re-runs the analyzer in memory and requires exact JSON equality.
