# AtlasRAG Benchmark and Evaluation Methodology

## Scope

The integrated package separates four evidence layers:

1. Neutral public retrieval quality and local systems measurements.
2. Clean installed-wheel AtlasRAG package evaluation.
3. Synthetic authorization and failure-contract evaluation.
4. Small integrated API smoke validation.

Results are attributed to the layer that produced them. HNSW belongs only to the neutral harness. Package BM25, exact dense, and RRF belong to AtlasRAG commit `5e86c78a4c40bc6d552d14d4fdcc370b0db8ece1` and wheel SHA-256 `30cbaf0030fe86177b7962e43267b6d182534c023eb9d61e7eec7481df048200`.

## Frozen datasets

### SciFact

- One whole BEIR corpus document per retrieval item.
- 300 judged test queries and 339 positive judgments.
- File-level rights are recorded in `DATASET_PROVENANCE.json`: ODC-By 1.0 for corpus abstracts, CC BY 4.0 for claim/evidence annotations, and Apache 2.0 for official repository code.
- The generic BEIR mirror label is recorded but does not replace the official component map.

### ArguAna contrast slice

- One whole BEIR argument passage per retrieval item.
- Deterministic 200-query slice from the 1,406-query test set.
- Selection: sort by `SHA256("atlasrag-arguana-contrast-20260731:" + query_id)`, then query ID; take 200.
- Identical query/document IDs are excluded before corpus statistics, scoring, and fusion when present. The frozen slice has 183 query IDs present in the corpus and 17 absent.
- This is a contrast slice, not a full official ArguAna score.
- The upstream Zenodo record and BEIR mirror expose conflicting license metadata, so the release candidate redistributes no ArguAna payloads.

### Synthetic fixtures

All text, tenant names, groups, shard IDs, failures, and unsupported-query cases are synthetic. No employer or private corpus material is used.

## Neutral retrieval methods

- BM25: dependency-light corpus-wide Okapi BM25, k1 1.5, b 0.75, stable external-ID ties.
- Exact dense: normalized `all-MiniLM-L6-v2` embeddings and exhaustive NumPy cosine search.
- HNSW: hnswlib cosine index, M 16, construction ef 160, search ef 100.
- Hybrid: RRF over top-100 BM25 and exact-dense candidates, RRF k 60.

## Installed AtlasRAG methods

The adapter creates one deterministic `Document` and one whole-document `Chunk` per corpus record, retaining an external-ID-to-chunk-ID mapping for qrels. It constructs:

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

The runner imports AtlasRAG from an installed wheel and rejects an import from the source tree. Package result rows preserve external ID, chunk ID, method, score kind, contribution ranks, source URI, document and content hashes, exact span, and chunking strategy.

## Model

- `sentence-transformers/all-MiniLM-L6-v2`
- Frozen cached revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Apache-2.0 model-card license
- Corpus and query vectors L2-normalized

## Metrics

Macro-averaged over judged queries:

- Recall@10
- MRR@10
- nDCG@10
- Success@10

Additional evidence:

- HNSW top-10 overlap with exact dense in the neutral harness
- sampled p50, p95, mean, and maximum latency
- model, corpus embedding, cache, and index construction timing
- package wheel and dataset hashes
- A/B quality and raw-ranking equality
- authorization and provenance contract gates

## Timing semantics

Neutral timing uses vectorized NumPy/HNSW paths. Installed timing uses AtlasRAG's dependency-light Python exact scorer and includes per-query embedding for dense and hybrid. Model load and index construction are excluded from both online latency summaries and recorded separately.

Because the implementation paths differ, neutral and installed latency values are not compared as if they were the same benchmark. All timing is single-host local evidence, not a service-level objective.

## Reproducibility

- Dataset inputs, model revision, seed, top-k, candidate-k, and RRF k are frozen.
- Neutral final evaluations run twice and require exact quality and raw-ranking equality.
- Installed-wheel evaluations run twice and require exact package identity, dataset hashes, quality, and raw-ranking equality.
- Installed-package A/B passed for every method on both datasets.
- Neutral-versus-package equivalence is informative, not a release gate. Score ties, deterministic identity tie breakers, and distinct numeric paths can change local ordering.

## Commands

See `RUNBOOK.md`. Machine-readable results live in `artifacts/installed-package-summary.json` and `artifacts/installed-package-reproducibility.json`.
