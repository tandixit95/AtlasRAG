# AtlasRAG

[![CI](https://github.com/tandixit95/AtlasRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/tandixit95/AtlasRAG/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/tandixit95/AtlasRAG)](https://github.com/tandixit95/AtlasRAG/releases/latest)

A permission-aware retrieval systems lab with source provenance, BM25, exact dense
retrieval, hybrid fusion, and evidence-gated optional reranking.

**The engineering question:** does an additional retrieval stage earn its latency
and complexity without weakening authorization, reproducibility, or quality?

Start with the [runnable access-boundary demo](examples/permission_boundary.py),
[rejected-reranker case study](benchmarks/promotion/RESULTS.md), or
[architecture](ARCHITECTURE.md). The full implementation and benchmark walkthrough
is preserved in [the technical guide](TECHNICAL_GUIDE.md).

## Try the actual code without model downloads

Python 3.11 or newer. From this repository checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/permission_boundary.py
python -m pytest tests/test_reviewer_demo.py
```

The demo indexes one public document and two protected documents. An anonymous
caller sees only the public document. A tenant-A operations caller sees the
public and tenant-A documents; a tenant-A caller in the wrong group sees only the
public document. No model, API key, network service, or private dataset is used.

This is a small demonstration backed by regression tests, not a security audit or
production-scale benchmark.

## Architecture and tradeoffs

```text
Document source -> immutable document -> chunk + provenance + access policy
                                                |
                         +----------------------+----------------------+
                         v                                             v
               permission-aware BM25                         exact dense search
                         +----------------------+----------------------+
                                                v
                                         reciprocal rank fusion
                                                v
                                   optional authorization-safe reranker
                                                v
                                    ranked results + citations + trace
```

Authorization is applied on every retrieval path. BM25 corpus statistics are
computed over the caller-visible corpus so hidden documents do not influence
visible scores. This favors an auditable security baseline over large-corpus
serving efficiency. Exact dense search is a correctness reference, not ANN.

## A negative result that changed the default decision

The frozen promotion protocol rejected the tested depth-10 reranker. The stored
A/B artifacts reproduced exactly, but a complete evidence package did not make
the candidate good enough to ship as the default.

| Stored evaluation | MRR@10 change | Paired 95% interval | Interpretation |
|---|---:|---|---|
| SciFact, 300 queries | +0.0139 | [-0.0149, +0.0420] | Improvement inconclusive |
| ArguAna, 200-query contrast slice | -0.0609 | [-0.1067, -0.0142] | Ordering regression |

Controlled reranker p95 also exceeded the frozen 75 ms component budget on both
tasks. Of 37 checks, six failed and none lacked evidence. **Hybrid RRF remains
the default; reranking remains opt-in.** The later frozen candidate evaluation
also retained the default; see [the current project state](PROJECT_STATE.md).

The ArguAna slice is not the full official benchmark. These are stored,
checksummed local experimental results, not production SLOs. Checking artifacts
is not the same as rerunning model inference. See [results](benchmarks/promotion/RESULTS.md),
[methodology](benchmarks/METHODOLOGY.md), and [limitations](benchmarks/LIMITATIONS.md).

## Scope and reproducibility

The latest stable release is `v0.2.0`; `main` contains `0.3.0.dev0` development
work. Core retrieval has no required third-party runtime dependency. Optional
Sentence Transformers adapters provide embedding and cross-encoder integration.

```bash
python -m pytest
ruff check .
ruff format --check .
python benchmarks/promotion/verify_artifacts.py
```

Only local plain-text ingestion is implemented. Distributed serving, persistence,
ANN indexing, generation, production traffic, and external adoption are not
claimed. The [scale evidence harness](benchmarks/scale) labels its large target
configuration as unexecuted; smoke tests are not large-scale retrieval evidence.

This repository is a new, public reconstruction of engineering ideas, not recovered
historical employer source. [Reconstruction history](RECONSTRUCTION_LEDGER.md),
[implementation details](TECHNICAL_GUIDE.md), and [claim boundaries](benchmarks/CLAIM_LEDGER.md)
remain available. This presentation/demo refresh was AI-assisted; passing automated
tests is not represented as independent human review.
