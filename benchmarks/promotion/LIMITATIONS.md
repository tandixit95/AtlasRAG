# Promotion Evaluation Limitations

- The evidence covers two benchmark task shapes, one dense model, one cross-encoder, one candidate depth, and one local GPU host.
- Whole BEIR documents are benchmark retrieval units; they are not AtlasRAG's default chunking strategy.
- The ArguAna result is a deterministic 200-query contrast slice, not a full official test-set score.
- The ArguAna license records conflict, so no dataset payload is redistributed.
- Controlled-host latency is a local component gate, not a production SLO or general hardware expectation.
- The CPU-load and exclusive-lock controls reduce known contention but do not provide dedicated bare-metal isolation.
- The experiment does not evaluate deeper candidate prefixes, model fine-tuning, generation quality, answer faithfulness, or end-user utility.
- A rejection under this frozen protocol does not prove that every reranker or every task will regress. It proves that this candidate did not satisfy these declared default-promotion requirements.
- Depth 20 remains rejected by the earlier dominated-result evidence and was not retested here.
- Depth 50 remains a quality-oriented option and was not a default-promotion candidate in this protocol.
- No production traffic, adoption, user count, deployment topology, service-level objective, or business impact is claimed.
