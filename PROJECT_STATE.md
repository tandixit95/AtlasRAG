# Project State

## Objective

Build AtlasRAG into a permission-aware, evaluation-driven retrieval system whose public claims are backed by executable tests, frozen public or synthetic data, reproducible measurements, and explicit limitations.

## Completed milestones

- Reconstruction foundation and public repository established.
- Immutable document model with logical identity and SHA-256 content versioning.
- Plain-text source adapter and source abstraction.
- Chunk domain model with exact source spans, source-version provenance, and immutable metadata.
- Deterministic fixed-character chunking control and ingestion pipeline.
- Embedding model contract and optional Sentence Transformers adapter.
- Exact exhaustive cosine retrieval correctness baseline.
- Shared typed `RetrievalQuery` and `RetrievalResult` contracts.
- Explicit score semantics for cosine, BM25, and RRF rather than cross-method score calibration.
- Dependency-light BM25 lexical retrieval.
- Reciprocal Rank Fusion over lexical and exact dense candidate rankings.
- Tenant and group permission policy encoded in chunk metadata.
- Fail-closed policy parsing and authorization filtering on dense, lexical, and hybrid paths.
- Authorization-scoped BM25 corpus statistics so invisible chunks do not perturb visible scores or ranks.
- Deterministic ranking and chunk-ID tie-breaking.
- Regression coverage for provenance, positive retrieval, no-match behavior, malformed policies, unauthorized leakage, stable ties, raw hybrid contribution metadata, and invalid embedding contracts.
- Public benchmark evidence for installed-package BM25, exact dense, and RRF across SciFact and a deterministic ArguAna contrast slice, with A/B ranking checks and explicit limitations.
- CI across Python 3.11-3.13 plus clean wheel and source-distribution install/import validation.
- Model-independent reranking contract and authorization-safe candidate composition.
- Optional cross-encoder adapter with configurable batching.
- Immutable citations plus rerank traces preserving candidate-stage evidence.
- Clean-wheel SciFact reranking experiment with exact A/B ranking reproduction, paired bootstrap analysis, raw timing samples, and an explicit no-promotion decision.
- Default-path evaluation standard and machine-readable depth-10 promotion protocol frozen before the second-task outcomes.
- Fail-closed promotion evaluator with explicit rejected, inconclusive, and promoted dispositions.
- Completed controlled A/B promotion evaluation across SciFact and the deterministic ArguAna contrast slice, with exact ranking reproduction, complete citations, and zero authorization leakage.
- Rejected depth-10 default promotion: ArguAna ordering quality regressed with paired intervals below zero, and controlled reranker p95 exceeded the frozen component budget on both tasks.

## Current architecture

```text
DocumentSource -> Document -> ChunkingStrategy -> Chunk
                                             |
                                             v
                                  PermissionPolicy metadata
                                             |
                 +---------------------------+--------------------------+
                 |                                                      |
                 v                                                      v
          BM25Retriever                                      ExactDenseRetriever
                 |                                                      |
                 +------------------- RRF rank fusion ------------------+
                                             |
                                             v
                                      RetrievalResult
                                             |
                                             v
                         RerankedRetriever + optional CrossEncoderReranker
                                             |
                                             v
                         RetrievalResult + Citation + RerankTrace
```

The core runtime remains standard-library-only. Sentence Transformers is an optional embedding extra. Exact dense search is the correctness reference and is not replaced by ANN in this milestone.

## Benchmark status

Version 0.2.0 publishes a compact in-repository evidence surface and a checksummed full archive with raw rankings. The installed-package evaluation covers 300 SciFact queries and a deterministic 200-query ArguAna contrast slice. A/B runs reproduced every quality metric and raw top-10 ranking for BM25, exact dense, and RRF. Authorization/provenance smoke tests returned zero unauthorized results across nine checks.

Current `main` adds a separate `0.3.0.dev0` reranking track and a frozen default-promotion standard. The original SciFact experiment found positive depth-10 point estimates but no paired interval excluding zero; depth 20 was dominated by depth 10, and depth 50 remained a quality-oriented option.

The Day 6 protocol then evaluated depth 10 on SciFact and the deterministic ArguAna contrast slice. All A/B rankings and quality summaries reproduced exactly, 5,000 published result citations were complete, and the installed-package safety harness observed zero authorization or exclusion leakage. SciFact MRR@10 remained inconclusive at `+0.0139` with interval `[-0.0149, +0.0420]`. ArguAna MRR@10 regressed by `-0.0609` with interval `[-0.1067, -0.0142]`, and nDCG@10 regressed by `-0.0476` with interval `[-0.0831, -0.0124]`. Controlled reranker p95 reproduced across A/B runs but exceeded the frozen 75 ms component budget on both tasks. The machine disposition is `retain_default_rejected`; hybrid RRF remains the default.

The evaluated wheels, source commits, data identities, model revisions, raw rankings, and checksums are pinned. Timings remain single-host controlled observations and are not production SLOs. HNSW remains neutral-harness evidence, not an AtlasRAG capability.

## Next highest-value task

Perform a no-payload failure analysis of the ArguAna rank movements and profile the reranker component on a separate development task. Any revised candidate must be tuned outside the frozen final sets, then locked under a new protocol before SciFact or ArguAna is re-evaluated. Do not add retrieval infrastructure merely to offset a failed promotion result.

## Deferred work

- HNSW or another ANN index, gated by corpus scale and exact-recall comparison.
- Structure-aware or tokenizer-aware chunking, gated by measured retrieval gains.
- Persistence and incremental index refresh.
- Generation.
- Explicit partial/degraded serving metadata for multi-index or remote failures.
- Context construction, abstention, and answer evaluation.
- Distributed serving, replication, global term statistics, and production SLOs.

## Known limitations

- Only local plain-text sources are supported.
- Chunking is character-based rather than tokenizer- or structure-aware.
- URI-derived document identity does not yet handle aliases or moves.
- BM25 recomputes visible corpus statistics per query. This strengthens the local security baseline but is not a scalable serving design.
- The tokenizer is a transparent Unicode word-token baseline, not a language-specific analyzer.
- Exact dense retrieval is exhaustive and intended for correctness, not large-corpus latency.
- Permission policy is conjunctive tenant/group metadata. It is not a general policy language or external authorization service.
- Reranking has two pinned task shapes for depth 10: SciFact improvement remains inconclusive, while the ArguAna contrast slice shows a ranking regression. Controlled component timing exceeds the frozen budget and is not a production SLO.
- No production traffic, user count, scale, SLO, or deployment topology is claimed.

## Publication status

The canonical repository is public at `tandixit95/AtlasRAG`; `main` is the canonical branch. Version 0.2.0 is the first evidence-backed GitHub release. Current `main` is `0.3.0.dev0` and adds an unreleased reranking/citation boundary plus a rejected default-promotion evidence package. No `v0.3.0` release exists.
