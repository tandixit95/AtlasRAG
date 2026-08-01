# AtlasRAG Benchmark Results

Status: AtlasRAG v0.2.0 evidence snapshot, 2026-08-01.

## Evidence tracks

This package preserves two result families instead of blending them:

1. A neutral NumPy/HNSW harness used for method comparison and ANN analysis.
2. Clean installed-wheel AtlasRAG runs for package BM25, exact dense retrieval, and RRF.

The frozen datasets, model revision, top-k, candidate-k, RRF constant, and metric code are aligned. Implementations and timing paths are not identical, so each table keeps its own attribution.

## Neutral harness quality and sampled latency

| Dataset | Method | Recall@10 | MRR@10 | nDCG@10 | Success@10 | p95 run A |
|---|---|---:|---:|---:|---:|---:|
| SciFact | BM25 | 0.7816 | 0.6340 | 0.6646 | 0.8000 | 1.52 ms |
| SciFact | Exact dense | 0.7833 | 0.6047 | 0.6451 | 0.7933 | 9.94 ms |
| SciFact | HNSW ANN | 0.7833 | 0.6047 | 0.6451 | 0.7933 | 1.17 ms |
| SciFact | Hybrid RRF | 0.8212 | 0.6467 | 0.6858 | 0.8367 | 22.26 ms |
| ArguAna-200 | BM25 | 0.7550 | 0.3562 | 0.4506 | 0.7550 | 12.37 ms |
| ArguAna-200 | Exact dense | 0.8100 | 0.4099 | 0.5044 | 0.8100 | 16.75 ms |
| ArguAna-200 | HNSW ANN | 0.8100 | 0.4099 | 0.5044 | 0.8100 | 1.24 ms |
| ArguAna-200 | Hybrid RRF | 0.8350 | 0.4343 | 0.5292 | 0.8350 | 41.78 ms |

Neutral A/B reruns reproduced quality and raw top-10 rankings exactly. HNSW is present only in this evidence track. Mean HNSW top-10 overlap with exact dense was 0.9967 on SciFact and 0.9990 on the ArguAna contrast slice.

## Installed AtlasRAG 0.2.0 quality and sampled latency

Package identity:

- Git commit: `5e86c78a4c40bc6d552d14d4fdcc370b0db8ece1`
- Wheel SHA-256: `30cbaf0030fe86177b7962e43267b6d182534c023eb9d61e7eec7481df048200`
- Import origin: installed wheel in site-packages, with the source tree rejected by the runner
- Methods: package `BM25Retriever`, `ExactDenseRetriever`, and `ReciprocalRankFusionRetriever`

| Dataset | Method | Recall@10 | MRR@10 | nDCG@10 | Success@10 | p95 A / B |
|---|---|---:|---:|---:|---:|---:|
| SciFact | BM25 | 0.7816 | 0.6340 | 0.6646 | 0.8000 | 48.39 / 43.43 ms |
| SciFact | Exact dense | 0.7833 | 0.6047 | 0.6451 | 0.7933 | 173.89 / 153.69 ms |
| SciFact | Hybrid RRF | 0.8212 | 0.6449 | 0.6845 | 0.8367 | 280.25 / 284.69 ms |
| ArguAna-200 | BM25 | 0.7600 | 0.3532 | 0.4494 | 0.7600 | 540.08 / 519.11 ms |
| ArguAna-200 | Exact dense | 0.8100 | 0.4028 | 0.4991 | 0.8100 | 447.15 / 367.89 ms |
| ArguAna-200 | Hybrid RRF | 0.8450 | 0.4288 | 0.5274 | 0.8450 | 896.13 / 762.13 ms |

Installed-package A/B reruns reproduced every quality metric and every raw top-10 ranking for all three methods on both datasets. The raw ranking JSONL and summary CSV are byte-identical within each A/B pair.

Package latency includes per-query embedding where applicable and the package's dependency-light Python exact scoring. It excludes model load and index construction. These values are not interchangeable with the vectorized neutral-harness latency values and do not support a production SLO.

## Installed versus neutral comparison

### SciFact

- Package BM25 and exact dense quality match the neutral harness exactly.
- Package and neutral top-10 rankings match on all 300 queries for BM25 and exact dense.
- Package hybrid matches neutral Recall@10 and Success@10, while MRR@10 is lower by 0.0017 and nDCG@10 by 0.0013.
- Package and neutral hybrid rankings match on 270 of 300 queries. The differences are bounded rank ordering changes around equal or nearly equal RRF scores.

### ArguAna deterministic 200-query contrast slice

- Package versus neutral top-10 ranking matches: BM25 170/200, exact dense 186/200, hybrid 155/200.
- Package BM25 Recall@10 is higher by 0.0050 while MRR@10 and nDCG@10 are lower by 0.0030 and 0.0012.
- Package exact-dense Recall@10 is equal; MRR@10 and nDCG@10 are lower by 0.0072 and 0.0052.
- Package hybrid Recall@10 and Success@10 are higher by 0.0100; MRR@10 and nDCG@10 are lower by 0.0055 and 0.0018.
- These are implementation-specific results on a fixed slice, not evidence that one implementation is universally superior.

## Authorization, provenance, and reliability

- Integrated synthetic AtlasRAG API smoke: nine authorization checks, zero unauthorized returns, provenance completeness 1.0, deterministic rerun true, and BM25 visible score/rank invariance after adding an unauthorized matching chunk.
- Neutral reliability harness: seven scenarios, zero unauthorized returns, complete provenance, and correct declared partial, stale, and unsupported signaling.
- Installed public benchmark artifacts preserve required AtlasRAG chunk provenance and external dataset ID mapping.

## Reproducibility artifacts

- `artifacts/installed-package-summary.json`
- `artifacts/installed-package-reproducibility.json`
- `artifacts/scifact-atlasrag-installed-a.json` and `-b.json`
- `artifacts/arguana-contrast-atlasrag-installed-a.json` and `-b.json`
- Per-query JSONL and summary CSV files next to each run artifact

## Negative and mixed findings retained

- BM25 beats exact dense on neutral SciFact nDCG@10, while exact dense beats BM25 on the ArguAna contrast slice.
- HNSW matches aggregate exact-dense quality but not every ranking.
- Package and neutral result families are close but not identical.
- Local timing varies materially and is much slower in the dependency-light package scorer than in the vectorized neutral harness.
- The ArguAna result remains a deterministic 200-query contrast slice, not the full official test score.
