# Promotion Evidence Claim Ledger

## Approved claims

- The depth-10 candidate was evaluated under machine-readable gates frozen before the second-task outcomes.
- Installed-wheel A/B runs reproduced SciFact and ArguAna rankings and aggregate quality exactly.
- Authorization, exclusion, malformed-policy, citation, and deterministic-rerun contracts passed.
- SciFact retained positive MRR@10 and nDCG@10 point estimates, but the paired MRR interval included zero.
- On the deterministic ArguAna contrast slice, depth-10 reranking reduced MRR@10 by 0.0609 and nDCG@10 by 0.0476; both paired 95% intervals were below zero.
- Controlled reranker p95 reproduced within the frozen cross-run ratio on both tasks but exceeded the frozen 75 ms budget.
- The machine decision is `retain_default_rejected`.
- Hybrid RRF remains the default; reranking remains opt-in.

## Required qualifiers

- ArguAna means the deterministic 200-query contrast slice, not a full official score.
- Latency values are controlled observations from one local RTX 4060 Laptop GPU, not production SLOs.
- The quality conclusion applies to the pinned package, model revisions, candidate depth, task shapes, and gate protocol.
- No third-party dataset text is redistributed.

## Forbidden claims

- Reranking is generally harmful.
- Hybrid RRF is universally superior.
- The candidate failed a production SLO.
- The latency values are general hardware expectations.
- The evaluation is a formal security certification.
- AtlasRAG has production users, traffic, adoption, or business impact.
- The reconstructed source is recovered historical code.
- A `v0.3.0` release exists.
