# AtlasRAG Architecture

## Current baseline

AtlasRAG currently stops at the ingestion/chunk boundary:

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
  - content_sha256: source-version fingerprint
  - metadata: immutable source metadata
     |
     v
ChunkingStrategy.chunk()
     |
     v
   Chunk
  - chunk_id: deterministic artifact identity
  - document_id + document_content_sha256: source/version lineage
  - start_char / end_char: exact half-open source span
  - content_sha256: chunk payload fingerprint
  - strategy_id: chunking implementation/config identity
  - metadata: immutable propagated source metadata
```

`IngestionPipeline` composes source loading and chunking while returning both the original `Document` objects and the ordered `Chunk` artifacts.

The next layer now provides a model-independent embedding contract and exact exhaustive cosine retrieval. There is still no lexical retrieval, fusion, ANN index, reranking, generation, or authorization layer.

## Decisions

### Keep the domain model independent of RAG frameworks

The ingestion layer uses only the Python standard library. LangChain, LlamaIndex, vector databases, tokenizer SDKs, and model clients are intentionally absent.

**Reason:** source, chunk, identity, and provenance contracts should remain stable while later experiments swap retrieval implementations. Dependencies should be introduced for demonstrated capability, not résumé decoration.

### Separate document identity from content version

`document_id` is deterministic for a normalized source URI. `content_sha256` is deterministic for the exact text payload.

**Reason:** a document edited in place remains the same logical source but becomes a new source version. This distinction supports change detection and prevents source identity from being conflated with payload identity.

**Tradeoff:** URI-derived identity assumes the URI is the canonical logical identifier. Source moves, aliases, and systems with external stable IDs will eventually need an explicit identity policy.

### Make chunks provenance-bearing domain objects

A chunk carries source URI, logical document ID, document content digest, exact character offsets, chunk digest, strategy ID, and immutable metadata.

**Reason:** retrieval results should not need to reverse-engineer where text came from. Provenance is established at transformation time and can be preserved through indexing, retrieval, reranking, and citation generation.

### Use stable chunk IDs without coupling them to unrelated document edits

Chunk identity is derived from the logical document ID, chunking strategy/configuration, character span, and chunk content digest. The whole-document digest is recorded as provenance but is not part of the chunk ID.

**Reason:** if text elsewhere in a source changes while a chunk's span and contents remain identical, that chunk can retain identity. This creates a path toward selective re-indexing or re-embedding rather than invalidating every chunk on every source edit.

**Consequence:** edits that shift offsets change downstream chunk IDs even when some text is repeated. That is deliberate for the current baseline because offsets are part of citation provenance. More sophisticated structural identity can be evaluated later if incremental ingestion requires it.

### Start with fixed-character chunking as a control

`FixedCharacterChunker` supports `chunk_size` and `overlap`, validates configurations, produces exact source slices, and stops when the final chunk reaches the source end.

**Reason:** this baseline is deterministic, dependency-free, and easy to audit. It gives future tokenizer-aware or structure-aware chunkers something measurable to beat.

**Tradeoff:** character windows ignore tokenization, sentence boundaries, headings, tables, and semantic structure. Those limitations are expected and should be addressed through controlled retrieval experiments rather than hidden by premature complexity.

### Character offsets refer to Python text, not encoded bytes

`start_char` and `end_char` are half-open indices into the decoded Python string.

**Reason:** chunk text can be verified directly with `document.text[start_char:end_char]`, including Unicode content. Byte offsets, PDF coordinates, timestamps, and other modality-specific locators can be added when corresponding source adapters exist.

### Preserve exact text at ingestion and chunking

The plain-text adapter does not trim or normalize source contents, and the fixed-character strategy slices the exact decoded string.

**Reason:** normalization is a transformation and should be explicit, testable, and attributable rather than silently changing source material.

### Snapshot metadata at domain boundaries

Document and chunk metadata are copied and exposed as read-only mappings.

**Reason:** downstream stages should not mutate provenance through shared dictionary references.

## Next architectural boundary

The next useful layer is retrieval, beginning with an embedding abstraction and a deliberately simple dense baseline. Before adding approximate indexes or framework integrations, the system should establish:

1. an embedding contract;
2. a minimal index/retriever contract;
3. top-k result semantics and scores;
4. a tiny frozen retrieval dataset suitable for regression tests;
5. deterministic tests around ranking and provenance preservation.

BM25, hybrid fusion, RRF, reranking, and larger evaluation experiments should build on that measurable baseline rather than arrive as one opaque stack.

## Failure modes already addressed

- invalid chunk size or overlap configurations;
- terminal overlap producing a redundant tail chunk;
- inconsistent text/content hashes in domain objects;
- provenance metadata mutation after object creation;
- loss of document source/version lineage during chunking;
- non-deterministic chunk identity for identical inputs/configuration.

## Failure modes to design for later

- duplicate logical sources under different URIs;
- source moves or aliases;
- changed content with stale chunks or embeddings;
- tokenizer/model changes invalidating stored embeddings;
- provenance loss between retrieval and generation;
- authorization filtering after retrieval instead of before/within candidate generation;
- benchmark leakage into tuning;
- quality gains that hide unacceptable latency or cost regressions;
- retrieval metrics that improve while answer faithfulness degrades.


## Dense retrieval baseline

`EmbeddingModel` separates embedding providers from retrieval semantics. `ExactDenseRetriever` embeds chunks, validates vector cardinality/dimensions, and performs exhaustive cosine search with deterministic chunk-ID tie breaking. Retrieval results retain the original `Chunk`, so source/version provenance survives ranking without a second lookup.

The first real adapter targets `sentence-transformers/all-MiniLM-L6-v2` as an optional dependency. Exact search is deliberately retained before HNSW: it is the small-corpus correctness reference against which future ANN recall/latency tradeoffs can be measured. No retrieval-quality benchmark is claimed yet.
