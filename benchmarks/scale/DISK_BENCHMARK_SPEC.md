# Disk-Backed Retrieval Benchmark Specification v1

## Status

This is a **frozen, unexecuted measurement protocol** for the next AtlasRAG scale milestone. It defines what must be implemented and measured before disk-backed retrieval, large-corpus latency, throughput, incremental updates, or authorization-at-scale can become public result claims. The machine-readable source of truth is `protocols/disk-backed-v1-unexecuted.json`.

No disk-backed index, 100K+ retrieval run, production QPS, production latency, or 100M execution is claimed by this specification.

## Corpus and identity

- Use only deterministic synthetic AtlasRAG records for this milestone; no employer, customer, or third-party corpus payloads.
- Scale order is 100K, 1M, 5M, then 10M documents. 100M is outside the v1 execution ladder and remains evaluation-only until 10M evidence exists.
- Document text size follows the frozen deterministic uniform integer-byte distribution from 512 through 2,048 bytes.
- Documents retain logical source identity plus SHA-256 content versioning.
- Chunks use the existing `fixed-character-v1:size=512:overlap=64` strategy and existing deterministic chunk identity contract.
- Every build must publish exact generated document count, chunk count, index record count, and physical index bytes.

## Authorization model

The benchmark uses the existing conjunctive tenant/group policy model. Public records have no protected metadata. Protected records require the matching tenant and at least one allowed group. Authorization filtering occurs **before scoring and ranking**. Malformed policy metadata fails closed.

Every measured bundle must report zero unauthorized results, zero excluded-chunk leakage, zero unauthorized BM25 influence failures, and complete citation/provenance fields. These are veto gates.

## Retrieval paths

The bounded disk-backed path must expose:

1. AtlasRAG BM25 with `k1=1.5`, `b=0.75`;
2. a named disk-backed dense implementation using pinned `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
3. Reciprocal Rank Fusion with `rrf_k=60` and candidate depth 100;
4. the current exhaustive dense retriever as the correctness reference for an audit sample.

The disk dense implementation name and exact dependency/version must be frozen before the first measured run. The protocol validator intentionally rejects a measured interpretation while that implementation field remains a pre-execution placeholder.

## Query distribution and concurrency

The frozen workload uses deterministic ordering and three query classes: 80% positive exact lookup, 15% authorization-negative, and 5% malformed-policy probes. Each concurrency point uses 200 warm-up queries followed by 2,000 measured queries. Concurrency levels are 1, 4, and 16. Each query has a 10-second timeout.

For every concurrency/cache condition, retain raw latency samples and report P50, P95, P99, sustained QPS, error count, and timeout count. Latency and QPS are result fields, not production SLOs. v1 deliberately defines no arbitrary performance-promotion threshold.

## Update distribution and recovery

The mixed replay is 90% reads, 6% updates, 2% inserts, and 2% deletes, with deterministic event identities and ordering. Checkpoint every 1,000 events. Required behavior includes incremental add, content update, delete, restart/resume, and idempotent replay.

An interrupted-and-resumed replay must produce the same final index record count, index checksum, and query-ranking hashes as an uninterrupted replay. Missing equivalence evidence is inconclusive; a mismatch fails the milestone.

## Cache conditions

Two conditions are required:

- **Cold cache:** fresh worker and index handle, with the OS page-cache reset method explicitly recorded. If the OS cache cannot be controlled or identified, the cold-cache result is inconclusive and must not be described as a storage-cold measurement.
- **Warm cache:** same worker/index handle after the declared 200-query warm-up.

This prevents a process restart alone from being mislabeled as a physical cold-cache measurement.

## Hardware and software disclosure

Every run records the AtlasRAG commit, package version, wheel SHA-256, Python version, OS, kernel, CPU model, physical/logical core counts, RAM bytes, storage model and medium, filesystem, free disk before/after, pinned dense model identity, and dependency-lock checksum. Storage serial numbers, credentials, tokens, and private local paths are forbidden from public artifacts.

## Measurements

Required build evidence: document/chunk/index counts, physical disk bytes, generation time, index build time, checkpoint count, and resume time.

Required query evidence: query count, concurrency, cache condition, raw latency samples, P50/P95/P99, sustained QPS, errors, and timeouts.

Required quality/safety evidence: BM25 expected-document Recall@10, disk-dense Recall@10 versus the exact reference, hybrid expected-document Recall@10, authorization leakage, excluded-chunk leakage, unauthorized influence checks, and citation completeness.

Required update evidence: applied inserts/updates/deletes, replay errors, final index count/checksum, and resume equivalence.

## Gates

A scale-specific bundle passes only when all complete gates pass:

- BM25 expected-document Recall@10 = 1.0;
- disk-dense Recall@10 versus exact reference >= 0.99;
- hybrid expected-document Recall@10 >= 0.99;
- citation completeness = 1.0;
- authorization leakage = 0;
- excluded-chunk leakage = 0;
- unauthorized influence failures = 0;
- malformed protected metadata fails closed;
- two equivalent runs reproduce ranking hashes and quality summaries;
- resumed execution is equivalent to uninterrupted execution;
- query error count = 0 and timeout count = 0.

A complete violated gate is a failure. Missing required evidence, contaminated measurement conditions, or an undeclared cache-reset method for a cold-cache claim is inconclusive. Neither failure nor inconclusive status supports a scale claim.

## Raw artifacts and checksums

Every measured run must produce `environment.json`, `build.json`, `query_summary.json`, `query_samples.csv`, `quality.json`, `authorization.json`, `updates.json`, `manifest.json`, and `SHA256SUMS`. The public package may contain IDs, ranks, timings, counts, hashes, and synthetic metadata, but must not redistribute third-party/private payload text or private local paths.

Two equivalent runs are required. Ranking hashes and quality summaries must match exactly; timing and environment fields are recorded but are not required to be byte-identical.

## Scale policy

- 100K may run locally only after the protocol validator, smoke evidence, and implementation-specific tests pass.
- 1M, 5M, and 10M each require a pre-run runtime/disk/RAM/GPU/cost estimate **and explicit approval before execution**, regardless of whether the run is local or paid.
- 100M is not executable under this protocol.

Target values never become measurements through prose.
