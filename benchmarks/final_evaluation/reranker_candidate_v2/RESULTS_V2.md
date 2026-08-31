# Reranker Candidate v2 Final Evaluation Results

## Decision

`retain_default_rejected` - the depth-10 / batch-32 cross-encoder candidate is **not** promoted. Hybrid RRF remains the default.

The candidate and all gates were committed before final outcomes were examined. The final evaluation used the exact frozen candidate from the synthetic development profile: candidate depth `10`, reranker batch size `32`.

## SciFact test-300

- MRR@10 mean delta: `+0.013935`
- MRR@10 paired 95% interval: `[-0.014944, +0.042049]`
- nDCG@10 mean delta: `+0.006819`
- Recall@10 mean delta: `+0.000000`
- Controlled reranker p95 A/B: `91.930 ms` / `95.269 ms`
- Frozen p95 budget: `75 ms` - **failed**
- Rankings and summary reproducibility: byte-identical across A/B

The positive MRR point estimate is not sufficient for promotion because the paired interval includes zero and the latency veto fails.

## ArguAna contrast-200

- MRR@10 mean delta: `-0.060935`
- MRR@10 paired 95% interval: `[-0.106692, -0.014188]`
- nDCG@10 mean delta: `-0.047641`
- nDCG@10 paired 95% interval: `[-0.083101, -0.012372]`
- Recall@10 mean delta: `+0.000000`
- Controlled reranker p95 A/B: `138.855 ms` / `135.481 ms`
- Frozen p95 budget: `75 ms` - **failed**
- Rankings and summary reproducibility: byte-identical across A/B

ArguAna remains a clear ranking-quality regression at the frozen candidate depth, so the candidate is rejected independently of the latency veto.

## Safety and reproducibility

- Authorization leakage: `0`
- Unauthorized candidate scoring: `0`
- Excluded-chunk leakage: `0`
- Malformed protected metadata: fails closed
- Citation completeness: `1.0`
- A/B ranking reproduction: exact for both task shapes
- Frozen batch-size veto: batch `32` verified in both task identities

## Limitations

These are controlled single-host component measurements, not production SLOs. The ArguAna evaluation uses the previously frozen deterministic contrast slice. No dataset payload is redistributed in this evidence package. The negative result does not imply that every reranker/model/depth would fail; it establishes only that this exact frozen candidate does not clear AtlasRAG's promotion gates.
