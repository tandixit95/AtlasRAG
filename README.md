# AtlasRAG

[![CI](https://github.com/tandixit95/AtlasRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/tandixit95/AtlasRAG/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/tandixit95/AtlasRAG)](https://github.com/tandixit95/AtlasRAG/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AtlasRAG is a reconstruction-first retrieval systems lab. It rebuilds prior Applied AI and RAG experience as a new public engineering artifact rather than presenting unavailable historical source as recovered code.

The latest stable release is `v0.2.0`. Current `main` advances toward `0.3.0` with a coherent, framework-independent retrieval and reranking slice:

- immutable documents and chunks with deterministic IDs, SHA-256 versioning, exact character spans, and source metadata;
- deterministic fixed-character chunking as an auditable control strategy;
- an embedding contract plus optional `sentence-transformers/all-MiniLM-L6-v2` adapter;
- exact exhaustive cosine retrieval as the dense correctness reference;
- a dependency-light BM25 lexical baseline;
- Reciprocal Rank Fusion (RRF) over BM25 and exact dense rankings;
- an authorization-safe reranking boundary plus an optional cross-encoder adapter;
- immutable source-span citations and rerank traces that preserve candidate-stage evidence;
- explicit tenant and group permission metadata enforced by every retrieval path;
- typed query and result contracts that preserve method, rank, score semantics, component contributions, and chunk provenance;
- deterministic tie-breaking and regression tests for authorization leakage, malformed policies, edge cases, and reproducibility;
- a frozen, fail-closed evaluation standard for deciding whether a candidate may replace the default retrieval path.

AtlasRAG publishes a bounded, reproducible `v0.2.0` benchmark evidence package for BM25, exact dense retrieval, and RRF. Reranking on `main` is not enabled by default and earns a public result claim only through a separately pinned experiment. AtlasRAG does not claim distributed serving, production traffic, model training, generation quality, formal security certification, or ANN scale.

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
       authorization-safe candidates
                  |
                  v
         RerankedRetriever (optional)
                  |
                  v
 RetrievalResult[]
 - original Chunk and immutable Citation
 - final method, rank, score, and score kind
 - raw component ranks/scores for hybrid results
 - candidate-stage method/rank/score in RerankTrace
```

## Permission model

A chunk is public when it has no AtlasRAG access metadata. Protected chunks use a `PermissionPolicy` with an optional tenant and optional allowed groups:

- tenant present: caller tenant must match;
- groups present: caller must belong to at least one allowed group;
- both present: both checks must pass;
- malformed access metadata fails during indexing instead of degrading to public access.

BM25 computes document frequency and length statistics only over chunks visible to the current principal. Unauthorized chunks therefore cannot appear in results or perturb the authorized caller's BM25 scores and ranks. Exact dense retrieval filters the candidate set before scoring, and hybrid retrieval fuses only already-authorized component results.

## Install

Python 3.11 or newer is required.

Latest stable release (`v0.2.0`, retrieval without reranking):

```bash
python -m pip install "atlasrag @ git+https://github.com/tandixit95/AtlasRAG.git@v0.2.0"
```

Current development branch (`0.3.0.dev0`, including reranking):

```bash
python -m pip install "atlasrag[reranking] @ git+https://github.com/tandixit95/AtlasRAG.git@main"
```

The core runtime has no required third-party dependency. The example below targets current `main`; pin model revisions for reproducible experiments.

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
    RerankedRetriever,
    CrossEncoderReranker,
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

# Optional: rerank only the already-authorized hybrid candidate set.
retriever = RerankedRetriever(
    hybrid,
    CrossEncoderReranker(),
    candidate_k=50,
)

results = retriever.search(
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
    print(result.rank, result.method.value, result.score, result.citation.source_uri)
    for component in result.contributions:
        print("  ", component.method.value, component.rank, component.score)
```

Raw BM25, cosine, RRF, and cross-encoder scores are not treated as mutually calibrated. RRF preserves component evidence, and reranking preserves the complete candidate-stage method, rank, score kind, score, and contributions in `RerankTrace`.

## Repository layout

```text
.
|-- .github/workflows/ci.yml
|-- ARCHITECTURE.md
|-- EVALUATION_STANDARD.md
|-- BENCHMARK_ADAPTER.md
|-- CHANGELOG.md
|-- PROJECT_STATE.md
|-- RECONSTRUCTION_LEDGER.md
|-- benchmarks/
|-- docs/releases/
|-- pyproject.toml
|-- src/atlasrag/
|   |-- embeddings/
|   |-- evaluation/
|   |-- ingestion/
|   |-- models.py
|   `-- retrieval/
|       |-- access.py
|       |-- bm25.py
|       |-- contracts.py
|       |-- dense.py
|       |-- hybrid.py
|       `-- reranking.py
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

To use the real MiniLM or cross-encoder adapters:

```bash
python -m pip install -e '.[embeddings]'
python -m pip install -e '.[reranking]'
```

The core runtime remains standard-library-only. Model-backed adapters are optional and loaded lazily.

## Reproducible benchmark evidence

The [benchmark package](benchmarks/README.md) separates installed AtlasRAG results from a neutral comparison harness and preserves dataset provenance, parameters, limitations, machine-readable summaries, A/B reproducibility checks, and authorization/provenance gates.

| Evaluation | BM25 Recall@10 | Exact dense Recall@10 | Hybrid Recall@10 |
|---|---:|---:|---:|
| SciFact, 300 judged queries | 0.7816 | 0.7833 | 0.8212 |
| ArguAna deterministic 200-query contrast slice | 0.7600 | 0.8100 | 0.8450 |

The ArguAna result is not a full official test-set score. Package timing is single-host local evidence, not a production SLO. HNSW measurements remain neutral-harness evidence and are not an AtlasRAG capability. See [results](benchmarks/RESULTS.md), [methodology](benchmarks/METHODOLOGY.md), [limitations](benchmarks/LIMITATIONS.md), and the [claim ledger](benchmarks/CLAIM_LEDGER.md).

[`BENCHMARK_ADAPTER.md`](BENCHMARK_ADAPTER.md) defines the stable package interface used by the installed-wheel evaluation.

### Reranking development evidence

Current `main` also publishes a separate [`0.3.0.dev0` reranking experiment](benchmarks/reranking/README.md) over 300 SciFact queries. Clean installed-wheel A/B runs reproduced every aggregate quality value and the complete raw ranking artifact exactly.

| Method | Recall@10 | MRR@10 | nDCG@10 | Success@10 |
|---|---:|---:|---:|---:|
| Hybrid RRF | 0.8212 | 0.6449 | 0.6845 | 0.8367 |
| Reranked depth 10 | 0.8212 | 0.6589 | 0.6914 | 0.8367 |
| Reranked depth 20 | 0.8144 | 0.6579 | 0.6881 | 0.8300 |
| Reranked depth 50 | 0.8272 | 0.6614 | 0.6943 | 0.8400 |

Every paired 95% bootstrap interval for the quality deltas includes zero. Depth 20 is dominated by depth 10 in this experiment, and timing was not stable across both runs because one run overlapped heavy host contention. Reranking therefore remains opt-in and disabled by default.

### Frozen default-promotion protocol

[`EVALUATION_STANDARD.md`](EVALUATION_STANDARD.md) and the [`promotion evidence package`](benchmarks/promotion/README.md) apply a fail-closed protocol frozen before the second-task outcomes. All 37 gates had complete evidence. The candidate was rejected: SciFact's MRR@10 delta remained inconclusive at `+0.0139` with a 95% interval of `[-0.0149, +0.0420]`; the ArguAna contrast slice regressed by `-0.0609` MRR@10 with an interval of `[-0.1067, -0.0142]`; and controlled reranker p95 exceeded the frozen 75 ms budget on both tasks. Hybrid RRF remains the default, and reranking remains opt-in.

## Reconstruction integrity

The original local source is unavailable. This repository contains new reconstruction work committed on its real development timeline. It does not recreate old commits, backdate history, or label reconstructed code as the lost original implementation.

See [`RECONSTRUCTION_LEDGER.md`](RECONSTRUCTION_LEDGER.md) for the evidence boundary, [`ARCHITECTURE.md`](ARCHITECTURE.md) for current design decisions, and [`PROJECT_STATE.md`](PROJECT_STATE.md) for the current frontier.
