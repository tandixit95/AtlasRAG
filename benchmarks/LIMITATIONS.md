# Limitations and Truth Boundary

## Evidence boundary

This package establishes local implementation and measurement evidence. It does not establish production traffic, users, multi-node operation, internet-scale search, formal security, a production SLO, customer outcomes, or employer deployment.

## Dataset and statistical limits

- SciFact and the deterministic 200-query ArguAna contrast slice cannot establish universal retrieval superiority.
- The ArguAna result is not a full official test-set score.
- Whole documents are retrieval units; no optimal chunking claim is supported.
- Metrics are point estimates without confidence intervals or significance tests.
- One embedding model and two task shapes are evaluated.

## Implementation limits

- AtlasRAG 0.2.0 implements BM25, dependency-light exact dense retrieval, and RRF. It does not implement HNSW.
- HNSW results are neutral-harness approximation evidence only.
- Neutral and installed package implementations use different tie breakers and numeric paths. Their close but non-identical rankings are retained.
- The package exact scorer converts and compares Python vectors for dependency-light correctness. It is not a scalable serving design.
- AtlasRAG has no distributed coordinator, persistence layer, stale-index protocol, or unsupported-query router. Those remain adapter-level contract demonstrations.

## Timing limits

- All timings come from one WSL2 laptop with local files and no network hop.
- Package latency and neutral latency are not directly comparable.
- Local scheduler, thermal state, CPU/GPU state, and concurrent load can change timing materially.
- Peak RSS is process-wide and not component-isolated.
- No latency number is a production SLO.

## Authorization and reliability limits

- Synthetic tenant/group tests are regression evidence, not a security audit or proof of non-interference.
- Identity providers, revocation, audit logging, authorization-aware caching, encryption, adversarial inputs, and network partitions are outside scope.
- Provenance completeness checks retrieval metadata, not generated-answer faithfulness.

## License and redistribution limits

- No third-party dataset archive, extracted record, model weight, or cache belongs in the public release candidate.
- SciFact has official file-level rights; a blanket dataset license is inaccurate.
- ArguAna source records conflict: the upstream Zenodo record metadata and BEIR mirror metadata show different Creative Commons terms.
- The conservative release path ships source instructions, checksums, attribution, slice logic, and aggregate metrics, not dataset payloads.

## Publication status

The benchmark evidence is publicly released with AtlasRAG v0.2.0. It is not peer reviewed, independently replicated, accepted by a venue, or assigned a DOI.
