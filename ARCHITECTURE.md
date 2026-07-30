# AtlasRAG Architecture

## Day 1 baseline

AtlasRAG begins with a narrow ingestion boundary:

```text
external source
     |
     v
DocumentSource.load()
     |
     v
  Document
  - document_id: logical source identity
  - source_uri: provenance anchor
  - text: exact ingested payload
  - content_sha256: content version fingerprint
  - metadata: immutable source metadata
```

There is no chunking, embedding, retrieval, reranking, generation, or authorization layer in the current implementation.

## Decisions

### Keep the domain model independent of RAG frameworks

The first layer uses only the Python standard library. LangChain, LlamaIndex, vector databases, and model SDKs are intentionally absent.

**Reason:** the document and provenance contracts should remain stable even if later experiments swap retrieval or orchestration implementations. This also makes dependency cost visible when a framework is eventually introduced.

### Separate document identity from content version

`document_id` is deterministic for a normalized source URI. `content_sha256` is deterministic for the exact text payload.

**Reason:** a document edited in place is still the same logical source but has a new content version. Future ingestion can use this distinction for idempotent updates, re-chunking, and selective re-embedding.

**Tradeoff:** URI-derived identity assumes the source URI is the canonical logical identity. Systems with source moves, aliases, or external stable IDs will need an explicit identity strategy rather than silently relying on paths.

### Preserve exact text at ingestion

The plain-text source does not trim or normalize content.

**Reason:** normalization is a transformation step and should be explicit, testable, and attributable. Ingestion should not silently change source material.

### Snapshot metadata

Document metadata is copied and exposed as a read-only mapping.

**Reason:** downstream stages should not be able to mutate provenance attached to an already-created document by retaining a reference to the caller's dictionary.

## Near-term evolution

Planned progression, gated by working tests and evidence:

1. ingestion pipeline and configurable chunking;
2. dense embeddings and a baseline retriever;
3. BM25 lexical retrieval;
4. hybrid fusion with Reciprocal Rank Fusion;
5. reranking and citation/provenance propagation;
6. frozen retrieval evaluation set and controlled benchmark;
7. CI, regression gates, architecture cleanup, and published results.

ACL-aware retrieval, sufficient-context gating, ANN/HNSW experiments, load testing, and ablations come after the baseline retrieval/evaluation loop is measurable.

## Failure modes to design for later

- duplicate sources under different URIs;
- changed content with stale chunks or embeddings;
- provenance loss between retrieval and generation;
- authorization filtering after retrieval instead of before/within candidate generation;
- benchmark leakage into tuning;
- quality gains that hide unacceptable latency or cost regressions;
- retrieval metrics that improve while answer faithfulness degrades.
