# Public Claim Ledger

Status: AtlasRAG v0.2.0 public evidence snapshot, 2026-08-01.

## Installed AtlasRAG implementation claims

| ID | Allowed claim | Evidence | Forbidden expansion |
|---|---|---|---|
| C1 | AtlasRAG 0.2.0 commit `5e86c78a4c40bc6d552d14d4fdcc370b0db8ece1` implements deterministic identity/provenance, permission-aware BM25, exact dense retrieval, RRF, fail-closed tenant/group policies, typed scores, and pre-ranking chunk exclusions | Source, docs, 63 passing tests | Production deployment; external adoption; recovered historical code |
| C2 | Wheel SHA-256 `30cbaf0030fe86177b7962e43267b6d182534c023eb9d61e7eec7481df048200` was installed into a clean environment and imported from site-packages | Installed-run artifacts | Public package registry release |
| C3 | Clean installed-wheel A/B benchmarks reproduced every quality metric and raw top-10 ranking for BM25, exact dense, and RRF on both frozen evaluations | `installed-package-reproducibility.json` | Independent reproduction; cross-host reproduction |
| C4 | AtlasRAG does not implement HNSW in version 0.2.0 | Package source and project state | Attribute neutral HNSW results to AtlasRAG |
| C5 | AtlasRAG is a new reconstruction and does not recover or backdate unavailable historical source | Reconstruction ledger | Original historical public implementation |

## Installed package quality claims

| ID | Allowed claim | Evidence | Forbidden expansion |
|---|---|---|---|
| P1 | SciFact package BM25: Recall 0.7816, MRR 0.6340, nDCG 0.6646, Success 0.8000 | `scifact-atlasrag-installed-a.json` | Universal lexical quality |
| P2 | SciFact package exact dense: Recall 0.7833, MRR 0.6047, nDCG 0.6451, Success 0.7933 | same | Scalable vector service |
| P3 | SciFact package hybrid: Recall 0.8212, MRR 0.6449, nDCG 0.6845, Success 0.8367 | same | Statistical significance; hybrid always wins |
| P4 | ArguAna-200 package BM25: Recall/Success 0.7600, MRR 0.3532, nDCG 0.4494 | `arguana-contrast-atlasrag-installed-a.json` | Full official ArguAna score |
| P5 | ArguAna-200 package exact dense: Recall/Success 0.8100, MRR 0.4028, nDCG 0.4991 | same | Dense always beats lexical |
| P6 | ArguAna-200 package hybrid: Recall/Success 0.8450, MRR 0.4288, nDCG 0.5274 | same | Benchmark leadership; universal superiority |
| P7 | Package latency is single-host, dependency-light Python scorer timing and is not neutral-harness latency or a production SLO | Installed artifacts and methodology | Cross-track latency speedup/slowdown percentage |

## Neutral research claims

| ID | Allowed claim | Evidence | Forbidden expansion |
|---|---|---|---|
| N1 | Neutral SciFact hybrid nDCG@10 is 0.6858; neutral ArguAna-200 hybrid nDCG@10 is 0.5292 | Neutral A artifacts | Package metrics without the neutral qualifier |
| N2 | Neutral HNSW aggregate quality matches exact dense with mean top-10 overlap 0.9967 on SciFact and 0.9990 on ArguAna-200 | Neutral artifacts | Identical rankings; million-scale ANN guarantee |
| N3 | Neutral A/B quality and raw rankings reproduce exactly | `official-reproducibility.json` | Independent replication |

## Cross-track comparison claims

| ID | Allowed claim | Evidence | Forbidden expansion |
|---|---|---|---|
| X1 | SciFact package and neutral BM25/exact-dense quality match exactly; hybrid differs slightly in MRR/nDCG due to bounded ordering differences | `installed-package-reproducibility.json` | All implementations equivalent |
| X2 | ArguAna package and neutral rankings match on 170/200 BM25, 186/200 dense, and 155/200 hybrid queries | same | Regression without inspecting metric direction |
| X3 | Package and neutral result families remain separately labeled | docs and artifacts | Average or merge the two result families |

## Authorization and reliability claims

| ID | Allowed claim | Evidence | Forbidden expansion |
|---|---|---|---|
| A1 | Integrated smoke completed nine authorization checks with zero unauthorized returns, complete required provenance, deterministic reruns, and BM25 hidden-document score/rank invariance | `atlasrag-adapter-smoke.json` | Security proof; penetration test |
| A2 | Neutral seven-scenario harness passed its declared authorization, partial, stale, unsupported, provenance, and reproducibility gates | `safety-evaluation.json` | AtlasRAG distributed-coordinator capability |

## License and release claims

| ID | Allowed claim | Evidence | Forbidden expansion |
|---|---|---|---|
| L1 | Official SciFact rights are file scoped: ODC-By 1.0 for abstracts, CC BY 4.0 for claim/evidence annotations, Apache 2.0 for code | Official SciFact LICENSE.md | One blanket SciFact license |
| L2 | ArguAna upstream Zenodo metadata and BEIR mirror metadata conflict; no payload redistribution is approved in this release candidate | provenance records | Silently select the more permissive label |
| R1 | The evidence package is published as a GitHub release asset for AtlasRAG v0.2.0 | GitHub release and checksums | Peer reviewed, independently replicated, deployed, or submitted |
| R2 | No DOI, paper submission, hosted demo deployment, or external adoption claim is made | release notes and limitations | DOI, acceptance, deployment, or adoption without evidence |

## Audit rule

Machine-readable raw output and exact package identity outrank narrative prose. When evidence differs, correct the prose instead of averaging the measurements into a prettier story.
