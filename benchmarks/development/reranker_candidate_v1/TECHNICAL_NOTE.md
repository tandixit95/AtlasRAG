# Freeze Before Outcomes: A Fail-Closed Reranker Candidate Protocol

Reranker tuning becomes difficult to defend once final evaluation outcomes influence which configuration is tried next. AtlasRAG's development protocol separates **candidate selection** from **final evaluation** so that a rejected final result cannot quietly become tuning data for the next attempt.

This note describes the protocol and its implementation boundary. It does **not** report a completed model-backed profile, natural-workload quality result, production latency, throughput, or default-path promotion.

## The problem

AtlasRAG previously evaluated a fixed reranking candidate and rejected default promotion after the frozen final protocol completed. The next engineering question is legitimate: can a different batching configuration make the reranker component cheaper without weakening evaluation discipline?

The unsafe approach would be to inspect the failed final-task outcomes, try alternatives, and repeatedly re-run those same final sets until something looks better. That turns the final evaluation into a development set.

The development protocol instead establishes a one-way boundary:

1. define candidate-selection rules before running them;
2. use deterministic synthetic inputs only;
3. nominate at most one configuration from that development evidence;
4. freeze the nominated candidate and final-evaluation gates in a new namespace;
5. only then inspect a new final evaluation outcome.

## What is frozen

`PROTOCOL.json` pins the complete v1 development surface:

- reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`;
- model revision: `c5ee24cb16019beea0893ab7796b1df96625c6b8`;
- deterministic synthetic seed: `20260817`;
- synthetic queries: `32`;
- candidate depths: `10`, `20`, `50`;
- batch sizes: `16`, `32`, `64`;
- warmups per configuration: `5`;
- measured iterations per configuration: `30`;
- peak accelerator-memory cap: `6144 MiB`;
- same-depth p95 tolerance: at most `1.10x` the best passing batch size;
- default candidate depth: `10`;
- maximum nominated candidates: `1`.

The verifier checks these values exactly rather than treating them as advisory defaults. A changed seed, matrix, gate, host control, claim boundary, or model identity invalidates the frozen v1 protocol.

## Synthetic inputs, not final-task payloads

`profile_reranker.py` generates its workload from deterministic templates and a fixed vocabulary. It does not load final SciFact or ArguAna evaluation payloads. The generated workload is also checked for prohibited final-dataset names before profiling continues.

That synthetic workload is intentionally narrow. It can answer component-level questions such as:

- does batching materially change reranker-only p95 under the same candidate depth?
- do returned score counts match the requested pair count?
- are all scores finite?
- does an identical repeated input preserve the same ordering?
- does the run stay within the frozen accelerator-memory gate?

It cannot establish retrieval quality on natural queries. That limitation is a design property, not missing evidence to be filled in later from the final sets.

## Fail-closed execution controls

The profiler requires several execution controls before a result can be considered valid development evidence:

- the exact pinned model snapshot must already exist locally;
- Hugging Face model resolution is forced into offline/local-only mode;
- a non-blocking exclusive profiler lock prevents concurrent profiler runs;
- CUDA timing is synchronized when an accelerator is available;
- every measured timing sample is preserved rather than only aggregates;
- environment metadata is recorded;
- score count, finiteness, repeated-order determinism, and memory gates must pass.

Host cleanliness is an external admission requirement as well. A technically runnable command is not automatically a valid benchmark run. If unrelated accelerator work makes the host contended, the correct output is **no measurement**, not a number with an optimistic caveat attached afterward.

## Candidate nomination

The protocol permits at most one nominated configuration, and only at the frozen default candidate depth of `10` unless a separately justified development protocol changes that rule.

The run as a whole must first satisfy the frozen host controls. Within an admitted run, a batch size remains eligible only if its score-shape, determinism, and memory checks pass. Its p95 must also remain within `1.10x` of the best passing batch size at the same depth. The profiler may then nominate no more than one batch configuration.

The executed synthetic profile nominated depth 10 / batch 32 under the precommitted selection rule. A nomination is still not a promotion. It only identifies the single candidate that may advance to the **new freeze-before-outcomes final evaluation** under `benchmarks/final_evaluation/reranker_candidate_v2/`.

## Reproducible verification available today

The protocol and implementation can be verified without executing the real model-backed profile:

```bash
python benchmarks/development/reranker_candidate_v1/verify_protocol.py
pytest -q benchmarks/development/reranker_candidate_v1/tests
```

The development tests cover deterministic synthetic workload generation, protocol validation, exclusive locking, score validation, ordering determinism, candidate selection, and fail-closed error cases. They are also included in AtlasRAG's default `pytest` discovery so CI does not silently omit them.

The real model-backed profile remains **unexecuted** as of this note. Therefore there is no published timing table, memory result, selected batch size, natural-workload quality result, or default-promotion claim here.

## Why this boundary matters

A benchmark is useful only if its decision process is auditable. Freezing candidate-selection rules before seeing final outcomes makes a later result easier to interpret:

- a positive outcome is not the product of repeated final-set tuning;
- a negative outcome remains informative rather than becoming another hidden hyperparameter search step;
- operational failures such as host contention produce an explicit blocked state instead of contaminated evidence;
- the public claim surface stays smaller than the underlying engineering work.

That is the intended progression for AtlasRAG: development evidence selects at most one candidate, a new protocol freezes that candidate and its gates, and final evaluation remains a one-way decision boundary.

## Limitations

- The synthetic workload is not representative evidence for retrieval quality or user traffic.
- Single-host component timing, once executed, will not be a production SLO.
- The protocol does not justify model replacement, default promotion, or deployment.
- No final SciFact or ArguAna payload is used for candidate selection.
- The model-backed synthetic profile has not yet been executed, so no configuration has been nominated.

See [`PROTOCOL.json`](PROTOCOL.json), [`README.md`](README.md), [`profile_reranker.py`](profile_reranker.py), and [`verify_protocol.py`](verify_protocol.py) for the executable definition.
