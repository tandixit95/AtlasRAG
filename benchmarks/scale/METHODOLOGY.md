# Methodology

## Evidence layers

1. **Target configuration:** design intent only; `execution_status` is
   `target_unexecuted` and execution is refused.
2. **Measured smoke:** validates deterministic generation, sharding, workload,
   checksums, schemas, and claim controls on a bounded corpus.
3. **Measured scale:** reserved for an actually executed larger run with the same
   evidence contract. The label alone is not sufficient; artifacts must validate.

## Synthetic records

Records are generated from SHA-256-derived integers over the frozen seed and ordinal.
No process-global random state is used. Each record contains deterministic source
identity, text length, topic, unique query token, shard, and optional tenant/group
policy. Text explicitly states that it is synthetic and contains no employer,
customer, or private corpus material.

## Sharding and workloads

Shard assignment is deterministic. The harness calculates a whole-stream digest and
one digest/count per shard without requiring full materialization. Query records name
an expected document via a unique synthetic token and carry the corresponding
principal. Update records select deterministic document IDs and version 2.

## Reproducibility

Each measured configuration runs twice. Exact equality is required for all
claim-bearing fields: configuration identity, counts, logical bytes, stream/shard
hashes, workload hashes, and materialized-file hash when present. Environment and
elapsed time are diagnostic and excluded from exact equality.

## Timing

Elapsed generation time is stored only to diagnose gross regressions. It is not a
throughput benchmark, retrieval latency, production SLO, or cross-host comparison.
A future serving benchmark requires isolated warm-up, steady-state windows,
concurrency control, percentile latency, error accounting, and declared hardware.
