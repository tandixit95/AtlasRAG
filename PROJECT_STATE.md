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
```

The core runtime remains standard-library-only. Sentence Transformers is an optional embedding extra. Exact dense search is the correctness reference and is not replaced by ANN in this milestone.

## Benchmark status

No integrated AtlasRAG benchmark result is published or claimed yet. The API is ready for a frozen benchmark adapter, but quality, latency, authorization leakage, and failure behavior must be measured and reported separately. See `BENCHMARK_ADAPTER.md`.

## Next highest-value task

Run the Lane 02 reproducible benchmark package against this branch using the frozen adapter contract, then let Lane 08 validate raw outputs, checksums, clean installation, privacy/IP boundaries, and documentation claims before integration into canonical `main`.

## Deferred work

- HNSW or another ANN index, gated by corpus scale and exact-recall comparison.
- Structure-aware or tokenizer-aware chunking, gated by measured retrieval gains.
- Persistence and incremental index refresh.
- Reranking and generation.
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
- No public benchmark result, production traffic, user count, scale, SLO, or deployment topology is claimed.

## Publication status

The canonical repository is public at `tandixit95/AtlasRAG`; `main` remains canonical. This branch is a local merge candidate only until the integration lane completes validation.
