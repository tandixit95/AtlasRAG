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

The benchmarked wheel is pinned by SHA-256. `benchmarks/SOURCE_EQUIVALENCE.json` verifies that the release runtime source tree is byte-identical to the evaluated runtime source. Timings remain single-host evidence and are not production SLOs. HNSW remains neutral-harness evidence, not an AtlasRAG capability.

## Next highest-value task

Freeze and run the reranking experiment against a clean `0.3.0.dev0` wheel. Compare hybrid versus cross-encoder reranking across candidate depths, record candidate recall ceilings and added latency, and retain negative findings. Do not enable reranking by default unless measured gains justify it.

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
- The released `v0.2.0` evidence does not cover reranking; `main` reranking remains experimental until a pinned clean-wheel experiment is published.
- No production traffic, user count, scale, SLO, or deployment topology is claimed.

## Publication status

The canonical repository is public at `tandixit95/AtlasRAG`; `main` is the canonical branch. Version 0.2.0 is the first evidence-backed GitHub release. Current `main` is `0.3.0.dev0` and adds an unreleased reranking/citation boundary.
