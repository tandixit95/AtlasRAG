# Reconstruction Ledger

This ledger separates supported historical facts from code rebuilt now and design choices introduced during the reconstruction. Unknowns remain unknown until evidence supports them.

## KNOWN

### Original purpose

- Prior work involved building RAG and context-engineering systems for Applied AI use cases.
- The original local source code is no longer available, so this repository is not presented as recovered historical code.
- The intended public reconstruction is a permission-aware RAG platform that can demonstrate retrieval quality, provenance, access control, context construction, evaluation, and systems tradeoffs.

### Surviving evidence

- A prior chat reported a sandbox-only AtlasRAG foundation and commit `5b32cb1`, with 7/7 tests passing.
- That sandbox path is not available in the current WSL environment, so neither its files nor commit history are imported or claimed as this repository's history.

### Historical implementation details and metrics

- No AtlasRAG-specific historical benchmark dataset, raw benchmark output, or source snapshot is available in the current environment.
- No historical latency, retrieval-quality, scale, user-adoption, production-usage, or SLO number is claimed in this repository.
- Specific historical framework, vector-store, chunking, fusion, reranking, or authorization choices remain unknown.

### Capability claims to prove publicly

The reconstruction is intended to produce evidence for professional work in:

- RAG and context engineering;
- retrieval evaluation and reliability;
- permission-aware retrieval;
- AI infrastructure and reproducible experimentation;
- architectural tradeoff analysis.

These are evidence goals, not claims that the public repository reproduces an employer production system.

## RECONSTRUCTED

Implemented from scratch on the current repository timeline:

- Python package and test structure;
- immutable `Document` model;
- deterministic logical document identity derived from source URI;
- SHA-256 content fingerprinting for version and change detection;
- `DocumentSource` ingestion contract;
- exact UTF-8 plain-text ingestion with source provenance;
- immutable `Chunk` model with exact source spans, chunk hashing, document-version lineage, and metadata propagation;
- deterministic `ChunkingStrategy` boundary and fixed-character control strategy;
- `IngestionPipeline` composition from source loading through ordered chunk production;
- embedding provider contract and optional Sentence Transformers adapter;
- exact exhaustive cosine retrieval baseline;
- shared typed query, result, method, score-kind, and component-contribution contracts;
- explicit caller principal and chunk permission policy;
- fail-closed tenant and group metadata parsing;
- dependency-light BM25 lexical retrieval;
- authorization-scoped BM25 statistics;
- Reciprocal Rank Fusion over BM25 and exact dense ranks;
- deterministic tie-breaking on every retrieval path;
- positive, negative, edge-case, provenance, determinism, invalid-contract, and authorization-leakage tests.

- model-independent reranking boundary over authorization-safe candidates;
- optional cross-encoder scoring adapter with explicit batching;
- immutable citation projections derived from original chunks;
- rerank traces preserving candidate-stage rank, method, score semantics, and hybrid contributions;
- regression coverage proving reranking cannot reintroduce unauthorized or excluded chunks;

## NEW

Design choices introduced during this rebuild, without claiming they existed historically:

- explicit separation between logical source identity and content version;
- immutable metadata snapshots at domain boundaries;
- a framework-free core so retrieval libraries and model adapters remain replaceable;
- deterministic chunk IDs based on logical source, strategy, span, and chunk contents;
- exact character-span provenance as the first citation primitive;
- exact dense search retained as the correctness reference before ANN optimization;
- a common result contract that labels each score's semantics;
- RRF selected to avoid treating BM25 and cosine scores as calibrated;
- hybrid results retain raw component ranks and scores separately;
- tenant and group policy metadata is conjunctive and fails closed;
- BM25 statistics are computed only over visible chunks to prevent unauthorized corpus influence in the local baseline;
- public chunks are visible to all callers, while protected chunks require explicit principal context;
- HNSW, default-enabled reranking, generation, persistence, and distributed serving remain gated by measured need.

## UNCERTAIN OR NOT YET PROVED

- original historical chunking rules and overlap;
- historical embedding models or vector-store configuration;
- historical lexical retrieval, fusion, or reranking parameters;
- historical access-control model;
- historical context sufficiency, abstention, or generation behavior;
- reranking quality and latency on a clean pinned package;
- ANN recall and latency tradeoffs;
- load profile, deployment topology, user count, traffic, SLOs, or production observability;
- whether the current policy model matches any prior production authorization design.

## Public proof target

Every major public capability should be backed by one or more of:

1. executable tests;
2. frozen public, permissively licensed, or synthetic evaluation data;
3. reproducible commands and raw machine-readable outputs;
4. checksums and environment capture;
5. architecture decisions with explicit tradeoffs;
6. security and reliability failure cases;
7. clear separation between reconstructed project measurements and employer production experience.

The repository must never infer historical facts from newly reconstructed behavior.
