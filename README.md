# AtlasRAG

AtlasRAG is a from-scratch reconstruction of prior Applied AI / RAG work, rebuilt as a public engineering artifact rather than presented as recovered historical source.

The project is intended to become a permission-aware RAG platform for controlled experiments in ingestion, retrieval, reranking, context construction, evaluation, reliability, and operational tradeoffs. The repository will earn those claims incrementally with working code, tests, and reproducible measurements.

## Current scope

Day 1 deliberately establishes only the foundation:

- a typed document model with deterministic source identity and content hashing;
- an ingestion source abstraction;
- a dependency-free UTF-8 plain-text source;
- unit tests for model invariants and ingestion behavior;
- architecture and reconstruction records.

Hybrid retrieval, vector search, BM25, RRF, reranking, chunking, ACL enforcement, evaluation, and benchmarking are intentionally **not implemented yet**.

## Why this shape?

A RAG system becomes difficult to trust when provenance, identity, and ingestion semantics are vague. AtlasRAG starts with those boundaries before adding retrieval machinery. A logical document identity is derived from its source URI, while a separate SHA-256 digest records the exact content version. That distinction lets future ingestion code detect updates without pretending every edit is a new logical source.

## Repository layout

```text
.
├── ARCHITECTURE.md
├── RECONSTRUCTION_LEDGER.md
├── pyproject.toml
├── src/
│   └── atlasrag/
│       ├── __init__.py
│       ├── models.py
│       └── ingestion/
│           ├── __init__.py
│           ├── base.py
│           └── text.py
└── tests/
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

See [`RECONSTRUCTION_LEDGER.md`](RECONSTRUCTION_LEDGER.md) for the evidence boundary and [`ARCHITECTURE.md`](ARCHITECTURE.md) for current design decisions.
