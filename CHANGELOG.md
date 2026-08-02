# Changelog

All notable public changes to AtlasRAG are documented here.

## [Unreleased]

### Added

- Model-independent reranking contract and authorization-safe `RerankedRetriever` composition.
- Optional Sentence Transformers cross-encoder adapter with configurable batching.
- Immutable source-span `Citation` projection on every retrieval result.
- Rerank traces preserving candidate-stage method, rank, score semantics, and hybrid contributions.
- Regression coverage for candidate-depth behavior, exclusions, authorization boundaries, deterministic ties, score validation, and citation integrity.
- Clean-wheel SciFact reranking evidence with exact A/B ranking reproduction, paired bootstrap intervals, raw ranking checksums, and host-contention diagnostics.
- Normative default-path evaluation standard, frozen machine-readable promotion gates, a generalized installed-package benchmark runner, and a fail-closed promotion evaluator.
- Controlled installed-wheel SciFact and ArguAna depth-10 A/B evidence with exact ranking reproduction, complete citation checks, safety-contract evaluation, and machine-readable gate reporting.

### Decisions

- Reranking remains opt-in. The frozen depth-10 promotion protocol returned `retain_default_rejected`: ArguAna MRR@10 and nDCG@10 regressed with paired intervals below zero, and controlled reranker p95 exceeded the 75 ms component budget on both task shapes.
- Hybrid RRF remains the default. Depth 20 remains rejected by its dominated result; depth 50 remains a quality-oriented configuration rather than a default candidate.

### Changed

- Development package version advanced to `0.3.0.dev0`; the latest stable release remains `0.2.0`.

## [0.2.0] - 2026-08-01

### Added

- Deterministic document and chunk identity with SHA-256 content versioning and exact source spans.
- Framework-independent ingestion, embedding, retrieval, and result contracts.
- Permission-aware BM25, exhaustive cosine retrieval, and Reciprocal Rank Fusion.
- Fail-closed tenant and group policy parsing with authorization applied before scoring and fusion.
- Pre-ranking chunk exclusions for benchmark tasks that place the query document in the corpus.
- Reproducible benchmark methodology, machine-readable results, A/B ranking checks, and explicit claim boundaries.
- GitHub Actions coverage for Python 3.11 through 3.13, linting, benchmark artifact verification, and clean wheel/source-distribution install tests.

### Security and integrity

- Added regression coverage for authorization leakage and unauthorized-document influence on BM25 statistics.
- Preserved original chunk provenance and component score semantics through every retrieval path.
- Completed full Git-history credential and privacy-pattern scans before release; no confirmed secret required purge or rotation.

### Limitations

- The release is a local retrieval core, not a distributed search service.
- Exact dense retrieval is a correctness reference, not a large-corpus serving claim.
- Benchmark timings are single-host measurements and are not production SLOs.
- HNSW results in the evidence archive belong to a neutral companion harness, not the AtlasRAG package.
