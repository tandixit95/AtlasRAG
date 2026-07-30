# Project State

## Objective

Build AtlasRAG into a permission-aware, evaluation-driven retrieval system whose public claims are backed by executable tests, controlled experiments, and reproducible results.

## Completed milestones

- Reconstruction foundation and public repository established.
- Immutable document model with logical identity and content versioning.
- Plain-text source adapter and source abstraction.
- Chunk domain model with exact source spans and version provenance.
- Chunking strategy boundary plus configurable fixed-character baseline.
- Ingestion pipeline composing source loading and chunking.
- Regression tests covering current ingestion and chunking contracts.

## Current architecture

`DocumentSource -> Document -> ChunkingStrategy -> Chunk`, composed by `IngestionPipeline`.

The implementation is dependency-free at runtime and stops before embedding/retrieval.

## Benchmark status

No retrieval-quality or performance benchmark has been published yet. There is not yet a retrieval implementation to benchmark.

## Next highest-value task

Establish the dense retrieval baseline: embedding contract, first real embedding implementation, minimal vector retrieval path, top-k semantics, provenance-preserving retrieval results, and a tiny frozen evaluation fixture for deterministic regression coverage.

## Unresolved questions

- Which embedding model offers the right quality/reproducibility/dependency tradeoff for the first public baseline?
- Whether the first vector baseline should be exact cosine search before HNSW, so ANN tradeoffs can be measured rather than assumed.
- What small public corpus/query set should seed retrieval regression tests without contaminating later evaluation work?

## Known limitations

- Only local plain-text sources are supported.
- Chunking is character-based rather than tokenizer- or structure-aware.
- URI-derived document identity does not yet handle aliases/moves.
- No persistence, incremental ingestion store, embeddings, retrieval, ACL enforcement, generation, or evaluation harness exists yet.

## Publication status

Public on `tandixit95/AtlasRAG`; `main` is the canonical branch.
