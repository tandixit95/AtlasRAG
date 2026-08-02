# Promotion Evaluation Methodology

## Question

May cross-encoder reranking at candidate depth 10 replace AtlasRAG's current permission-aware hybrid RRF default?

## Frozen protocol

The protocol and evaluator were committed before the second-task outcomes were generated.

- Protocol commit: `c59485d698c41797dc307b81fa8a4198f1113812`
- Gate SHA-256: `ff186c5cd42839478d7b3e7f40377383cec2d7c472b5194eeeb4e70484c217c4`
- Baseline: hybrid RRF, top 10
- Candidate: cross-encoder reranking, candidate depth 10, final top 10
- Primary metric: MRR@10
- Secondary metric: nDCG@10
- Recall boundary: Recall@10 mean delta must be non-negative
- Bootstrap: 10,000 deterministic paired query-level resamples

The gate file was not relaxed after outcomes were visible. No final-task tuning was performed.

## Installed package and models

The evaluation imported AtlasRAG `0.3.0.dev0` from the same installed wheel evaluated during Day 5.

- Candidate implementation commit: `43c4ef33b212869c94ff8cd9bb1c8615b0084b24`
- Wheel SHA-256: `43b09b21f813f99f4b8c78d43a358c18a667dbc477da06b4ad92b3a312f8c928`
- Dense model: `sentence-transformers/all-MiniLM-L6-v2`
- Dense revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Reranker revision: `c5ee24cb16019beea0893ab7796b1df96625c6b8`

Both models were loaded from pinned local snapshots with offline mode enabled.

## Task shapes

### SciFact

- Official test split
- 300 judged queries
- 5,183 corpus documents
- Whole documents used only as benchmark retrieval units

### ArguAna contrast slice

- Deterministic 200-query slice already defined by the `v0.2.0` evidence program
- Selection: sort by `SHA256(atlasrag-arguana-contrast-20260731: + query_id)`, then query ID; take 200
- 8,674 corpus documents
- If a query ID is also a corpus ID, exclude that document before corpus statistics, scoring, fusion, and reranking
- No ArguAna payload is redistributed because upstream license metadata conflicts

## Execution

For each task:

1. Build one provenance-bearing AtlasRAG chunk per corpus document.
2. Load the frozen MiniLM corpus embedding cache.
3. Generate exactly 10 authorization-safe hybrid RRF candidates.
4. Score only those 10 candidates with the pinned cross-encoder.
5. Sort by reranker score, then prior candidate rank, then chunk ID.
6. Evaluate the final top 10.
7. Execute independent runs A and B.
8. Require byte-identical rankings and summary CSV files.

Because candidate depth equals final cutoff, reranking cannot alter Recall@10 or Success@10 unless the candidate set itself changes. The experiment tests ordering quality and component cost, not deeper candidate recall.

## Safety and citation contracts

A separate installed-package synthetic evaluation exercised:

- unauthorized-result leakage;
- unauthorized candidate scoring;
- pre-ranking exclusions;
- malformed protected metadata failure;
- authorized private visibility;
- deterministic reruns;
- complete immutable citations.

Every contract passed.

## Controlled-host timing

Each run:

- acquired an exclusive benchmark lock;
- ran one benchmark process at a time;
- loaded models offline from pinned snapshots;
- passed CPU-load preflight and latency-phase checks;
- recorded 25 deterministic latency queries;
- synchronized CUDA around measured GPU work;
- reported raw samples, p50, p95, and maximum.

The frozen reranker-component budget was p95 <= 75 ms. Cross-run p95 ratio had to be <= 1.25, and each run had to satisfy the frozen dispersion heuristic. These are controlled local component gates, not production SLOs.

## Decision semantics

- Any observed veto violation: `retain_default_rejected`.
- Missing evidence or unproven quality improvement without a veto: `retain_default_inconclusive`.
- Only all gates passing: `promote`.

The checker exits nonzero for both retain decisions. An explicit publication override permits writing a failed-closed report without changing the decision.
