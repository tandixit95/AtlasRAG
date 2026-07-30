# Reconstruction Ledger

This ledger separates remembered or supported historical facts from code rebuilt now and improvements introduced during the reconstruction. Unknowns stay unknown until evidence supports them.

## KNOWN

### Original purpose

- Prior work involved building RAG / context-engineering systems for Applied AI use cases.
- The original local source code is no longer available, so this repository is not presented as recovered historical code.
- The intended public reconstruction is a permission-aware RAG platform that can eventually demonstrate retrieval quality, provenance, access control, context construction, evaluation, and systems tradeoffs.

### Surviving evidence

- The prior chat reported a sandbox-only AtlasRAG foundation and commit `5b32cb1`, with 7/7 tests passing.
- That sandbox path is not available in the current WSL environment, so neither the files nor that commit are being imported or claimed as this repository's history.

### Historical implementation details and metrics

- No AtlasRAG-specific historical benchmark dataset, raw benchmark output, or source snapshot is available in the current environment.
- No historical latency, retrieval-quality, scale, user-adoption, or production-usage numbers are claimed in this repository at Day 1.
- Specific prior framework choices for AtlasRAG are treated as unknown unless surviving evidence is recovered later.

### Relevant capability claims to prove publicly

The reconstruction is intended to produce evidence for professional work in:

- RAG and context engineering;
- retrieval evaluation and reliability;
- permission-aware retrieval;
- AI infrastructure and reproducible experimentation;
- architectural tradeoff analysis.

These are goals for evidence, not claims that Day 1 already proves the final system.

## RECONSTRUCTED

Implemented from scratch in the current reconstruction:

- Python package and test structure;
- immutable `Document` model;
- deterministic logical document identity derived from source URI;
- SHA-256 content fingerprinting for version/change detection;
- `DocumentSource` ingestion contract;
- plain-text file ingestion with explicit source provenance;
- unit coverage for model invariants and text-source behavior;
- reproducible Python project configuration;
- immutable `Chunk` model with exact source spans, chunk hashing, document-version lineage, and metadata propagation;
- deterministic `ChunkingStrategy` boundary and configurable fixed-character chunking baseline;
- `IngestionPipeline` composition from source loading through ordered chunk production;
- regression coverage for chunk boundaries, overlap, identity, provenance, Unicode offsets, and end-to-end ingestion.

## NEW

Design choices introduced during this rebuild, without claiming they existed historically:

- explicit separation between logical source identity and content version;
- immutable metadata snapshots on documents to reduce accidental mutation across pipeline stages;
- a deliberately framework-free ingestion boundary so retrieval libraries can be evaluated later rather than baked into the domain model;
- deterministic chunk IDs that preserve identity for an unchanged source span/configuration even when unrelated document content changes;
- explicit chunk strategy IDs so artifacts remain attributable to the transformation configuration that produced them;
- exact character-span provenance as the first citation primitive, with richer modality-specific locators deferred until corresponding source types exist.

## UNCERTAIN / UNKNOWN

Not reconstructed or historically verified yet:

- original historical chunking rules and overlap (the current fixed-character strategy is reconstructed/new work);
- embedding models and vector-store configuration;
- lexical retrieval implementation;
- fusion and reranking parameters;
- access-control model;
- context sufficiency / abstention logic;
- evaluation dataset and acceptance thresholds;
- load profile, deployment topology, and observability design.

## Public proof target

As the repository matures, every major capability should be backed by one or more of:

1. executable tests;
2. frozen evaluation data;
3. reproducible benchmark commands and raw outputs;
4. architecture decisions with tradeoffs;
5. security/reliability failure cases and regression coverage.

The repository should never infer historical facts from newly reconstructed behavior.
