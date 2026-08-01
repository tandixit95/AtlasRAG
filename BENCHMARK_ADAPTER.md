# Benchmark Adapter Contract

This note defines the stable retrieval API for reproducible benchmark harnesses. It does not itself contain or authorize benchmark claims.

## Frozen adapter surface

Use these three methods from the installed AtlasRAG package:

```python
from atlasrag.retrieval import (
    BM25Retriever,
    ExactDenseRetriever,
    ReciprocalRankFusionRetriever,
    RetrievalQuery,
)

lexical = BM25Retriever(k1=1.5, b=0.75)
dense = ExactDenseRetriever(embedder)
hybrid = ReciprocalRankFusionRetriever(
    lexical,
    dense,
    rrf_k=60,
    candidate_k=100,
)

hybrid.index(chunks)  # indexes the same frozen chunk tuple in both components
```

For separate method measurements, call:

```python
lexical.search(RetrievalQuery(text=query_text, top_k=top_k, principal=principal))
dense.search(RetrievalQuery(text=query_text, top_k=top_k, principal=principal))
hybrid.search(RetrievalQuery(text=query_text, top_k=top_k, principal=principal))
```

Do not compare or average raw `result.score` values across methods. Interpret score using:

- `result.method`;
- `result.score_kind`;
- `result.contributions` for hybrid component ranks and raw scores.

For evaluation tasks that place the query document inside the corpus, map the external document ID to its AtlasRAG chunk ID and pass it through `RetrievalQuery(excluded_chunk_ids=...)`. Exclusions are applied before BM25 statistics, dense scoring, and hybrid fusion.

Use `result.chunk.chunk_id` as the retrieval item ID and preserve these provenance fields in raw output:

- `document_id`;
- `document_content_sha256`;
- `source_uri`;
- `start_char` and `end_char`;
- `content_sha256`;
- `strategy_id`.

## Required fixed inputs

The benchmark run should freeze and record:

- corpus and query source, license, version, and checksum;
- chunking strategy and exact parameters;
- embedding model name and immutable revision when available;
- BM25 `k1` and `b`;
- RRF `rrf_k` and `candidate_k`;
- `top_k` used for each metric;
- principal and permission fixture semantics;
- Python, package, OS, CPU, and relevant accelerator environment;
- warmup and latency measurement procedure;
- random seeds for any external dependency that uses them.

## Required result separation

Report these independently:

1. retrieval quality by method;
2. end-to-end query latency by method;
3. authorization leakage checks by method;
4. deterministic rerun equality for result IDs and quality metrics;
5. failures and excluded queries;
6. build/index time and peak memory when measured.

Do not fold authorization or failure tests into one quality aggregate. Do not call local single-host measurements distributed or production scale.

## Permission fixture

At minimum, include public, same-tenant, wrong-tenant, same-tenant allowed-group, and same-tenant wrong-group chunks. Assert zero unauthorized chunk IDs for BM25, exact dense, and hybrid paths.

For BM25, also compare the authorized result scores and ranks with and without an unauthorized matching chunk. They should remain unchanged because statistics are scoped to the visible corpus.

## Validation expectation

Run benchmark adapters from a clean installed package and emit raw machine-readable records using the fields above. Verify package identity, checksums, deterministic IDs, result equality, documentation, and privacy/IP boundaries before freezing any public claim.
