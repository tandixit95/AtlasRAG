# AtlasRAG

AtlasRAG is a from-scratch reconstruction of prior Applied AI / RAG work, rebuilt as a public engineering artifact rather than presented as recovered historical source.

The project is becoming a permission-aware, evaluation-driven RAG platform for controlled experiments in ingestion, retrieval, reranking, context construction, reliability, and systems tradeoffs. Capabilities are added only when the repository can prove them with working code, tests, or reproducible measurements.

## Current capabilities

The repository currently provides a framework-independent ingestion boundary:

- immutable `Document` objects with deterministic logical source identity and SHA-256 content versioning;
- `DocumentSource` adapters, including exact UTF-8 plain-text ingestion;
- immutable `Chunk` objects with source-version provenance, character offsets, configuration identity, and content hashes;
- a `ChunkingStrategy` contract;
- a deterministic fixed-character baseline with configurable overlap;
- an `IngestionPipeline` that preserves loaded documents and ordered derived chunks;
- regression coverage for identity, hashing, overlap boundaries, metadata propagation, Unicode offsets, empty sources, and end-to-end text ingestion.

Dense retrieval, BM25, hybrid fusion, reranking, ACL enforcement, generation, and benchmark claims are intentionally not implemented yet.

## Why this shape?

A retrieval system becomes difficult to trust when source identity, transformation boundaries, and provenance are vague. AtlasRAG makes those contracts explicit before adding retrieval machinery.

A logical document ID is derived from its source URI, while a separate digest identifies the exact document contents. Chunks carry both their own content hash and the source-document version that produced them. Chunk IDs are deterministic for a source, strategy configuration, character span, and chunk contents, allowing an unchanged span to retain identity even when unrelated content elsewhere in the source changes.

The first chunker is intentionally simple. Fixed-character windows are a transparent control configuration, not a claim that character chunking is optimal. Later strategies should earn their complexity through retrieval measurements.

## Data flow

```text
DocumentSource
     |
     v
  Document
     |
     v
ChunkingStrategy
     |
     v
   Chunk[]

IngestionPipeline composes the two stages and returns both
source documents and the ordered chunk artifacts derived from them.
```

## Repository layout

```text
.
├── ARCHITECTURE.md
├── PROJECT_STATE.md
├── RECONSTRUCTION_LEDGER.md
├── pyproject.toml
├── src/
│   └── atlasrag/
│       ├── __init__.py
│       ├── models.py
│       └── ingestion/
│           ├── __init__.py
│           ├── base.py
│           ├── chunking.py
│           ├── pipeline.py
│           └── text.py
└── tests/
    ├── test_chunking.py
    ├── test_ingestion_pipeline.py
    ├── test_models.py
    └── test_text_source.py
```

## Local development

Python 3.11+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

## Reconstruction integrity

The original local source is unavailable. This repository contains new reconstruction work committed on its real development timeline. It does not recreate old commits, backdate history, or label reconstructed code as the lost original implementation.

See [`RECONSTRUCTION_LEDGER.md`](RECONSTRUCTION_LEDGER.md) for the evidence boundary, [`ARCHITECTURE.md`](ARCHITECTURE.md) for current design decisions, and [`PROJECT_STATE.md`](PROJECT_STATE.md) for the current development frontier.
