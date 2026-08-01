from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import hnswlib
import numpy as np
from run_batched_benchmark import (
    MODEL_NAME,
    BM25Index,
    load_dataset,
    mean_metrics,
    metrics,
    ranked_from_scores,
    rrf,
)
from sentence_transformers import SentenceTransformer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    dataset = load_dataset("scifact", args.data_dir, "train")
    doc_ids = [str(d["_id"]) for d in dataset.corpus]
    corpus_texts = [
        (str(d.get("title", "")) + "\n" + str(d.get("text", ""))).strip()
        for d in dataset.corpus
    ]
    bm25 = BM25Index.build(doc_ids, corpus_texts)
    embeddings = np.load(args.workdir / "corpus_embeddings.npy", mmap_mode="r")
    model = SentenceTransformer(MODEL_NAME)
    query_embs = model.encode(
        [dataset.queries[qid] for qid in dataset.eval_ids],
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    exact_matrix = query_embs @ np.asarray(embeddings).T

    candidate_values = [10, 25, 50, 100, 200]
    rrf_values = [10, 60, 100]
    ann_ef_values = [10, 50, 100, 200]
    max_candidate = max(candidate_values)
    top_k = 10

    grid = {(c, rk): [] for c in candidate_values for rk in rrf_values}
    bm_quality = []
    exact_quality = []
    bm_rankings: list[list[str]] = []
    exact_rankings: list[list[str]] = []

    for row_idx, qid in enumerate(dataset.eval_ids):
        bm = bm25.search(dataset.queries[qid], max_candidate)
        exact = ranked_from_scores(exact_matrix[row_idx], doc_ids, max_candidate, None)
        bm_rankings.append(bm)
        exact_rankings.append(exact)
        bm_quality.append(metrics(bm, dataset.qrels[qid], top_k))
        exact_quality.append(metrics(exact, dataset.qrels[qid], top_k))
        for candidate_k in candidate_values:
            for rrf_k in rrf_values:
                fused = rrf(bm[:candidate_k], exact[:candidate_k], rrf_k, candidate_k)
                grid[(candidate_k, rrf_k)].append(
                    metrics(fused, dataset.qrels[qid], top_k)
                )

    rows = []
    for (candidate_k, rrf_k), values in sorted(grid.items()):
        rows.append(
            {
                "kind": "hybrid_grid",
                "candidate_k": candidate_k,
                "rrf_k": rrf_k,
                "ann_ef": None,
                **mean_metrics(values),
                "mean_top10_overlap": None,
            }
        )

    for ef in ann_ef_values:
        index = hnswlib.Index(space="cosine", dim=int(embeddings.shape[1]))
        index.load_index(str(args.workdir / "dense.hnsw"))
        index.set_ef(ef)
        labels, _ = index.knn_query(query_embs, k=top_k)
        quality = []
        overlaps = []
        for row_idx, qid in enumerate(dataset.eval_ids):
            ann = [doc_ids[int(i)] for i in labels[row_idx]]
            exact = exact_rankings[row_idx]
            quality.append(metrics(ann, dataset.qrels[qid], top_k))
            overlaps.append(len(set(ann) & set(exact[:top_k])) / top_k)
        rows.append(
            {
                "kind": "ann_ef",
                "candidate_k": top_k,
                "rrf_k": None,
                "ann_ef": ef,
                **mean_metrics(quality),
                "mean_top10_overlap": statistics.fmean(overlaps),
            }
        )

    payload = {
        "schema_version": "atlasrag.ablation.v2",
        "dataset": "scifact",
        "split": "train",
        "query_count": len(dataset.eval_ids),
        "purpose": "exploratory engineering ablations only; official test labels were not used for parameter selection",
        "fixed_test_configuration": {"candidate_k": 100, "rrf_k": 60, "ann_ef": 100},
        "baselines": {
            "bm25": mean_metrics(bm_quality),
            "exact_dense": mean_metrics(exact_quality),
        },
        "rows": rows,
        "interpretation": [
            "Candidate-k and RRF-k are explored on the SciFact train split, never the official test split.",
            "The final test configuration remains candidate_k=100 and rrf_k=60; stronger exploratory cells are reported but not used for a post-hoc test rerun.",
            "ANN ef is evaluated against exact dense top-10 overlap and retrieval quality; exact dense remains the oracle.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = args.output.with_suffix(".csv")
    fields = [
        "kind",
        "candidate_k",
        "rrf_k",
        "ann_ef",
        "recall@10",
        "mrr@10",
        "ndcg@10",
        "success@10",
        "mean_top10_overlap",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
