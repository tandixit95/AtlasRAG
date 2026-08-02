# Default-Path Promotion Evidence

This namespace implements the normative rules in `../../EVALUATION_STANDARD.md`.

The frozen protocol evaluates cross-encoder reranking at candidate depth 10 against the current hybrid RRF default on two task shapes:

- SciFact test, 300 judged queries;
- the existing deterministic ArguAna 200-query contrast slice.

`PROMOTION_GATES.json` was committed before the new task-shape outcomes were generated. The final evidence package must preserve that file byte-for-byte. Public artifacts contain identifiers, metrics, hashes, citation fields, and timing samples, but no query or corpus text.

The evaluator fails closed. A non-promoted candidate produces a nonzero exit code unless the caller explicitly uses the evidence-publication override.
