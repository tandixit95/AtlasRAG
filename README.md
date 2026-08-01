# AtlasRAG

AtlasRAG is a reconstruction-first retrieval systems lab. It rebuilds prior Applied AI and RAG experience as a new public engineering artifact rather than presenting unavailable historical source as recovered code.

The current implementation provides a coherent, framework-independent retrieval slice:

- immutable documents and chunks with deterministic IDs, SHA-256 versioning, exact character spans, and source metadata;
- deterministic fixed-character chunking as an auditable control strategy;
- an embedding contract plus optional `sentence-transformers/all-MiniLM-L6-v2` adapter;
- exact exhaustive cosine retrieval as the dense correctness reference;
- a dependency-light BM25 lexical baseline;
- Reciprocal Rank Fusion (RRF) over BM25 and exact dense rankings;
- explicit tenant and group permission metadata enforced by every retrieval path;
- typed query and result contracts that preserve method, rank, score semantics, component contributions, and chunk provenance;
- deterministic tie-breaking and regression tests for authorization leakage, malformed policies, edge cases, and reproducibility.

AtlasRAG does not yet claim a public benchmark result, distributed serving, production traffic, model training, generation quality, or ANN scale. Those claims must be earned by separate reproducible evidence.

## Retrieval flow

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
   Chunk + provenance + access metadata
     |
     +-------------------------+
     |                         |
     v                         v
BM25Retriever          ExactDenseRetriever
     |                         |
     +------------+------------+
                  |
                  v
      ReciprocalRankFusionRetriever
                  |
                  v
 RetrievalResult[]
 - original Chunk and provenance
 - method and rank
 - method-specific score + score kind
 - raw component ranks/scores for hybrid results
```

## Permission model

A chunk is public when it has no AtlasRAG access metadata. Protected chunks use a `PermissionPolicy` with an optional tenant and optional allowed groups:

- tenant present: caller tenant must match;
- groups present: caller must belong to at least one allowed group;
- both present: both checks must pass;
- malformed access metadata fails during indexing instead of degrading to public access.

BM25 computes document frequency and length statistics only over chunks visible to the current principal. Unauthorized chunks therefore cannot appear in results or perturb the authorized caller's BM25 scores and ranks. Exact dense retrieval filters the candidate set before scoring, and hybrid retrieval fuses only already-authorized component results.

## Minimal example

```python
from atlasrag.embeddings.base import EmbeddingModel
from atlasrag.ingestion import FixedCharacterChunker
from atlasrag.models import Document
from atlasrag.retrieval import (
    AccessPrincipal,
    BM25Retriever,
    ExactDenseRetriever,
    PermissionPolicy,
    ReciprocalRankFusionRetriever,
    RetrievalQuery,
)

# Supply any EmbeddingModel implementation. Deterministic test embedders are used
# in the test suite; SentenceTransformerEmbedding is available via [embeddings].
embedder: EmbeddingModel = ...

public = Document.from_text(
    source_uri="memory://public-guide",
    text="Mars rover operations guide",
)
private = Document.from_text(
    source_uri="memory://tenant-a-runbook",
    text="Mars rover incident runbook",
    metadata=PermissionPolicy(
        tenant_id="tenant-a",
        allowed_groups=frozenset({"ops"}),
    ).to_metadata(),
)

chunker = FixedCharacterChunker(chunk_size=500)
chunks = tuple(chunker.chunk(public)) + tuple(chunker.chunk(private))

lexical = BM25Retriever()
dense = ExactDenseRetriever(embedder)
hybrid = ReciprocalRankFusionRetriever(lexical, dense)
hybrid.index(chunks)

results = hybrid.search(
    RetrievalQuery(
        text="mars incident",
        top_k=5,
        principal=AccessPrincipal(
            tenant_id="tenant-a",
            groups=frozenset({"ops"}),
        ),
    )
)

for result in results:
    print(result.rank, result.method.value, result.score, result.chunk.source_uri)
    for component in result.contributions:
        print("  ", component.method.value, component.rank, component.score)
```

Raw BM25 and cosine scores are intentionally not added or treated as calibrated. RRF combines component ranks; the original component rank, score, and score kind remain attached for inspection.

## Repository layout

```text
.
|-- ARCHITECTURE.md
|-- BENCHMARK_ADAPTER.md
|-- PROJECT_STATE.md
|-- RECONSTRUCTION_LEDGER.md
|-- pyproject.toml
|-- src/atlasrag/
|   |-- embeddings/
|   |-- ingestion/
|   |-- models.py
|   `-- retrieval/
|       |-- access.py
|       |-- bm25.py
|       |-- contracts.py
|       |-- dense.py
|       `-- hybrid.py
`-- tests/
```

## Local development

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
```

To use the real MiniLM adapter:

```bash
python -m pip install -e '.[embeddings]'
```

The core runtime remains standard-library-only. The embedding extra is optional and loaded lazily.

## Benchmark boundary

This code milestone does not publish retrieval-quality or latency numbers. [`BENCHMARK_ADAPTER.md`](BENCHMARK_ADAPTER.md) defines the fixed API, inputs, result separation, authorization checks, and reproducibility requirements that future public benchmark evidence must satisfy. Raw outputs and limitations must accompany any result claim.

## Reconstruction integrity

The original local source is unavailable. This repository contains new reconstruction work committed on its real development timeline. It does not recreate old commits, backdate history, or label reconstructed code as the lost original implementation.

See [`RECONSTRUCTION_LEDGER.md`](RECONSTRUCTION_LEDGER.md) for the evidence boundary, [`ARCHITECTURE.md`](ARCHITECTURE.md) for current design decisions, and [`PROJECT_STATE.md`](PROJECT_STATE.md) for the current frontier.
