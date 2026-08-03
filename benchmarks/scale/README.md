# AtlasRAG Scale-Evidence Reconstruction

This directory rebuilds scale evidence from scratch. It does **not** recover the lost
historical project, backdate development, or treat a target configuration as a
measurement.

## Current milestone

The first milestone proves the evidence machinery rather than a large-scale serving
claim:

- deterministic synthetic documents with no employer/customer data;
- deterministic tenant/group policies and public/private distribution;
- deterministic shard assignment and per-shard SHA-256 digests;
- deterministic query and update workloads;
- optional corpus materialization with stream/file digest equality;
- two-run exact reproduction over the claim-bearing surface;
- fail-closed validation that rejects `target_unexecuted` as measured evidence;
- machine-readable schemas and explicit allowed/forbidden claims.

The committed smoke configuration executes 5,000 generated documents. The
`target-100m-unexecuted.json` file is a planning target only. Its presence supports
**no** claim that 100 million documents were generated, indexed, searched, or served.

## Run

```bash
python -m benchmarks.scale.src.run_scale_evidence \
  --config benchmarks/scale/configs/smoke-v1.json \
  --output benchmarks/scale/artifacts/smoke-v1.json

python -m benchmarks.scale.src.verify_scale_evidence \
  benchmarks/scale/artifacts/smoke-v1.json
```

To materialize a measured corpus, set `materialize_corpus` to `true` in a measured
configuration and pass `--materialized-dir`. Large corpora and third-party data must
not be committed.

## Next milestones

1. Add a disk-backed shard/index adapter with bounded local measurements.
2. Add a process-isolated query driver and explicit warm-up/measurement windows.
3. Add mixed read/update workload replay and authorization leakage gates.
4. Execute progressively larger measured configurations on declared hardware.
5. Publish a scale claim only after raw artifacts, checksums, environment details,
   reruns, and limitations support the exact wording.

See `METHODOLOGY.md`, `LIMITATIONS.md`, and `CLAIM_LEDGER.md`.
