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
- Embedding model contract and optional Sentence Transformers adapter.
- Exact exhaustive cosine retriever with deterministic top-k ranking and provenance-preserving results.

## Current architecture

`DocumentSource -> Document -> ChunkingStrategy -> Chunk`, composed by `IngestionPipeline`, followed by `EmbeddingModel -> ExactDenseRetriever -> RetrievalResult`.

Core runtime remains standard-library-only; the real Sentence Transformers adapter is optional.

## Benchmark status

No retrieval-quality or performance benchmark has been published yet. Exact dense retrieval now exists as a correctness baseline; benchmark/evaluation results remain deliberately unclaimed until a frozen dataset and reproducible experiment are added.

## Next highest-value task

Add lexical BM25 retrieval and a common result contract suitable for comparing dense and lexical candidates before introducing Reciprocal Rank Fusion. Preserve the exact dense path as the correctness baseline.

## Unresolved questions

- Whether `all-MiniLM-L6-v2` remains the first benchmark model after the frozen evaluation corpus is chosen.
- When ANN/HNSW becomes justified by corpus scale; exact cosine remains the current correctness reference.
- What small public corpus/query set should seed retrieval evaluation without contaminating later tuning.

## Known limitations

- Only local plain-text sources are supported.
- Chunking is character-based rather than tokenizer- or structure-aware.
- URI-derived document identity does not yet handle aliases/moves.
- No persistence, incremental ingestion store, lexical/hybrid retrieval, ANN index, ACL enforcement, generation, or evaluation harness exists yet.

## Publication status

Public on `tandixit95/AtlasRAG`; `main` is the canonical branch.
