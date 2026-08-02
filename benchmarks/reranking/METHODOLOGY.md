# Reranking Methodology

## Question

Does a pinned cross-encoder improve AtlasRAG hybrid top-10 rankings enough to justify its candidate-depth and latency cost?

## Package boundary

The benchmark imported AtlasRAG `0.3.0.dev0` from a clean installed wheel, not the source checkout.

- Commit: `43c4ef33b212869c94ff8cd9bb1c8615b0084b24`
- Wheel SHA-256: `43b09b21f813f99f4b8c78d43a358c18a667dbc477da06b4ad92b3a312f8c928`

## Dataset

SciFact test split:

- 300 judged queries
- 5,183 corpus documents
- whole documents used as benchmark retrieval units
- official file-level rights remain documented in `../DATASET_PROVENANCE.json`
- no third-party dataset text is redistributed here

The compressed ranking artifact contains query IDs, corpus IDs, metrics, ranks, scores, hashes, source URIs, and character spans. It contains no query or corpus text.

## Retrieval and reranking

1. Build one provenance-bearing AtlasRAG chunk per SciFact document.
2. Load the frozen MiniLM corpus embedding cache used by the `v0.2.0` benchmark.
3. Generate hybrid candidates with BM25, exact dense cosine retrieval, and RRF.
4. Score candidate prefixes of 10, 20, and 50 with the pinned cross-encoder.
5. Sort by cross-encoder score, then prior candidate rank, then chunk ID.
6. Evaluate the final top 10.

Candidate recall is reported because it is an upper bound: reranking cannot recover a relevant document absent from the candidate prefix.

## Models

Dense model:

- `sentence-transformers/all-MiniLM-L6-v2`
- revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Apache-2.0

Reranker:

- `cross-encoder/ms-marco-MiniLM-L6-v2`
- revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`
- Apache-2.0

Both model revisions were loaded from local immutable snapshots with offline mode enabled.

## Quality metrics

- Recall@10
- MRR@10
- nDCG@10
- Success@10
- candidate recall at depths 10, 20, and 50

A paired query-level analysis compares reranked results with the hybrid baseline. Ten thousand deterministic bootstrap resamples estimate 95% intervals for mean metric deltas. This bootstrap analysis was added after the frozen A/B runs and is not presented as preregistered inference.

## Reproducibility

Runs A and B used the same package, data identities, model revisions, parameters, seed, and latency-query IDs.

Accepted reproducibility gates:

- aggregate quality equality
- candidate-recall equality
- byte-identical raw ranking files
- byte-identical quality summary CSV files
- matching package, dataset, model, and configuration identities

Timing equality is not a gate. Scheduler and host load are uncontrolled variables.

## Latency

The runner records raw samples for:

- hybrid candidate generation
- reranker-only scoring
- directly measured end-to-end candidate generation plus reranking

Model load and index construction are excluded. CUDA synchronization brackets measured GPU work.

Run A overlapped substantial local agent CPU activity and showed extreme dispersion. Run B was comparatively stable. Therefore run-B latency is retained as a single-run observation, but no stable cross-run latency claim is approved.
