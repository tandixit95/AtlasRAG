# AtlasRAG Evaluation and Default-Path Promotion Standard

## Status and scope

This document is normative for replacing an AtlasRAG default retrieval path. It governs benchmark-backed default changes; it is not a production service-level objective, a formal security certification, or a claim that benchmark tasks represent all workloads.

The current default path is permission-aware hybrid retrieval using BM25, exact dense retrieval, and Reciprocal Rank Fusion. A candidate path remains opt-in until a frozen evidence package passes every required gate.

## Core rule

**A candidate may replace the default only when complete, frozen evidence proves that it is authorization-safe, provenance-complete, reproducible, non-regressive on recall, better on the declared primary ranking metric across the required task shapes, and inside a repeatable controlled-host latency budget. Missing, contradictory, contaminated, or statistically inconclusive evidence retains the current default.**

## Freeze-before-outcomes protocol

Before evaluating a new frozen task shape, commit the following:

1. candidate and baseline identities;
2. package, dataset, model, and configuration identities;
3. required task shapes and query-selection rules;
4. final cutoff and candidate depth;
5. primary and secondary quality metrics;
6. bootstrap procedure and seed policy;
7. authorization, citation, reproducibility, recall, quality, and latency gates;
8. failure semantics.

After outcomes are visible, the frozen gate file may not be relaxed for that candidate. A changed gate is a new evaluation protocol and requires a new evidence namespace. Final frozen evaluation data may not be used for tuning the candidate.

## Required evidence

Each required task shape must provide:

- two independently executed installed-package runs;
- byte-identical query-level rankings and quality summaries across the two runs;
- exact package, source commit, wheel hash, dataset hashes, model revisions, and parameters;
- per-query baseline and candidate rankings without redistributed query or corpus text;
- paired query-level Recall@K, MRR@K, nDCG@K, and Success@K deltas;
- deterministic bootstrap intervals for the declared primary metric;
- complete immutable citation fields for every published candidate result;
- controlled-host latency observations from both runs.

A separate synthetic contract evaluation must exercise authorization filtering, exclusions, malformed-policy failure, deterministic reruns, and citation completeness through the actual AtlasRAG API.

## Gate classes

### Veto gates

Any observed violation rejects the candidate for default promotion:

- authorization leakage count is not zero;
- excluded-chunk leakage count is not zero;
- malformed protected metadata does not fail closed;
- citation completeness is below 100 percent;
- A/B ranking or quality reproduction fails;
- dataset payload text appears in public evidence;
- candidate depth or package/model identity differs from the frozen protocol;
- Recall@K falls below the frozen no-regression boundary on any required task;
- controlled-host requirements fail;
- component latency exceeds the frozen budget or does not reproduce within the allowed ratio.

Missing veto evidence is **inconclusive**, not a pass.

### Promotion gates

After all veto gates pass, every required task shape must also satisfy:

- positive mean improvement on the declared primary ranking metric;
- a paired 95 percent bootstrap interval whose lower bound is strictly above zero for that primary metric;
- no negative mean delta on the declared secondary ranking metric.

Failure to establish these improvements is **inconclusive** unless a veto gate also fails.

## Controlled-host latency

Controlled-host evidence is a local component budget, not a production SLO. A qualifying run must:

- hold an exclusive benchmark lock;
- load models from pinned local snapshots with offline mode enabled;
- execute one benchmark process at a time;
- pass the frozen CPU-load preflight and latency-phase checks;
- report raw samples, p50, p95, and maximum values;
- avoid the frozen high-dispersion condition;
- reproduce p95 within the frozen cross-run ratio.

Timing from a run that fails these controls remains visible but cannot satisfy the latency gate.

## Decision semantics

The machine evaluator emits exactly one disposition:

- `promote`: every required gate passes;
- `retain_default_rejected`: complete evidence contains at least one veto violation;
- `retain_default_inconclusive`: evidence is missing or the required quality improvement is not established.

Only `promote` permits a default-path change. Both retain dispositions fail closed. The evaluator exits nonzero for either retain disposition unless an explicit evidence-publication flag is supplied.

## Current candidate boundary

The first protocol under this standard evaluates cross-encoder reranking at candidate depth 10 against the hybrid RRF default. Depth 20 remains rejected by the prior SciFact experiment unless separately frozen evidence reverses its dominated result. Depth 50 remains a quality-oriented configuration and is not the default candidate in this protocol.
