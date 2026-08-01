# Installed AtlasRAG Benchmark Adapter

Status: completed and validated for AtlasRAG 0.2.0.

## Bound package

- Commit: `5e86c78a4c40bc6d552d14d4fdcc370b0db8ece1`
- Version: 0.2.0
- Wheel SHA-256: `30cbaf0030fe86177b7962e43267b6d182534c023eb9d61e7eec7481df048200`
- Methods: `BM25Retriever`, `ExactDenseRetriever`, `ReciprocalRankFusionRetriever`
- Query contract: `RetrievalQuery`, including `excluded_chunk_ids`

## Corpus and identity mapping

Each public corpus record becomes one deterministic AtlasRAG `Document` and one whole-document `Chunk`. The adapter retains both external-dataset-ID to chunk-ID and chunk-ID to external-dataset-ID mappings so qrels are evaluated without discarding AtlasRAG identity and provenance.

ArguAna query-document self matches are excluded before BM25 corpus statistics, dense scoring, and RRF whenever the query ID exists in the corpus.

## Fixed construction

```python
lexical = BM25Retriever(k1=1.5, b=0.75)
dense = ExactDenseRetriever(embedder)
hybrid = ReciprocalRankFusionRetriever(
    lexical,
    dense,
    rrf_k=60,
    candidate_k=100,
)
hybrid.index(chunks)
```

The embedder uses `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## Output contract

Every retained result records:

- external dataset ID and deterministic chunk ID;
- rank, method, method-specific score, and score kind;
- hybrid contribution methods, component ranks, raw component scores, and score kinds;
- document ID and document content hash;
- source URI, exact span, chunk content hash, and strategy ID.

Raw BM25 and cosine scores are never summed. RRF is rank based.

## Installation guard

The runner records wheel hash, commit, version, and import origin. It fails if AtlasRAG imports from the forbidden source tree instead of the installed distribution.

## Acceptance evidence

All acceptance criteria passed:

1. Frozen dataset hashes match the provenance record.
2. Model revision and parameters are frozen.
3. Method and score semantics validate.
4. External-ID mapping is complete.
5. Required provenance is complete.
6. Query self-exclusion occurs before scoring where applicable.
7. A/B package identity, quality, and rankings match exactly.
8. Package artifacts are separate from neutral-harness artifacts.
9. Neutral/package differences are reported rather than normalized away.

Evidence:

- `artifacts/installed-package-summary.json`
- `artifacts/installed-package-reproducibility.json`
- `artifacts/scifact-atlasrag-installed-a.json` and `-b.json`
- `artifacts/arguana-contrast-atlasrag-installed-a.json` and `-b.json`
- `src/run_installed_atlasrag_benchmark.py`
- `src/check_installed_package_results.py`

## Capability boundary

HNSW, distributed coordination, stale-index handling, and unsupported-query routing are not AtlasRAG 0.2.0 package capabilities. They remain separately labeled neutral or adapter-level evidence.
