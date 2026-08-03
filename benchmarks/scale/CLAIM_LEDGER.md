# Scale-Evidence Claim Ledger

| ID | Allowed claim | Required evidence | Forbidden expansion |
|---|---|---|---|
| S1 | AtlasRAG has a deterministic synthetic scale-evidence harness | Source and tests | Large-scale execution |
| S2 | The smoke configuration generated 5,000 logical documents and deterministic workloads | Validated `artifacts/smoke-v1.json` | Retrieval or serving throughput |
| S3 | The smoke claim-bearing surface reproduced exactly across two executions | Reproducibility digests in the bundle | Independent or cross-host replication |
| S4 | The harness fails closed when an unexecuted target is presented as measured evidence | Unit tests and validator | Proof that all future reporting is error-free |
| T1 | A 100M-document configuration exists as an unexecuted target | `configs/target-100m-unexecuted.json` | 100M documents generated/indexed/queried |

## Audit rule

Target configuration values never become measurements through prose. A claim must
name the measured configuration, validated artifact, environment, and exact boundary.
When evidence is missing, remove or narrow the claim.
