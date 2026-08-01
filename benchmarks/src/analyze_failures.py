from __future__ import annotations

import argparse
import json
from pathlib import Path

METHODS = ("bm25", "exact_dense", "ann_hnsw", "hybrid_rrf")


def analyze(path: Path, dataset: str) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures = {m: [] for m in METHODS}
    for row in rows:
        for method in METHODS:
            metric = row["metrics"][method]
            success_key = next(k for k in metric if k.startswith("success@"))
            if metric[success_key] == 0:
                failures[method].append(row["query_id"])
    rescues = {
        "hybrid_over_bm25": sorted(set(failures["bm25"]) - set(failures["hybrid_rrf"])),
        "hybrid_over_exact_dense": sorted(
            set(failures["exact_dense"]) - set(failures["hybrid_rrf"])
        ),
        "hybrid_regressions_vs_bm25": sorted(
            set(failures["hybrid_rrf"]) - set(failures["bm25"])
        ),
        "hybrid_regressions_vs_exact_dense": sorted(
            set(failures["hybrid_rrf"]) - set(failures["exact_dense"])
        ),
    }
    examples = {}
    by_id = {row["query_id"]: row for row in rows}
    for label, ids in rescues.items():
        examples[label] = [
            {
                "query_id": qid,
                "relevant_ids": by_id[qid]["relevant_ids"],
                "top": by_id[qid]["top"],
            }
            for qid in ids[:5]
        ]
    return {
        "dataset": dataset,
        "query_count": len(rows),
        "failure_counts": {m: len(ids) for m, ids in failures.items()},
        "failure_query_ids": failures,
        "rescue_and_regression_counts": {k: len(v) for k, v in rescues.items()},
        "examples": examples,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scifact", type=Path, required=True)
    p.add_argument("--arguana", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = {
        "schema_version": "atlasrag.failure-analysis.v1",
        "datasets": [
            analyze(args.scifact, "scifact"),
            analyze(args.arguana, "arguana-contrast-200"),
        ],
        "limitations": [
            "A failure means no judged relevant item appeared in the top 10; it is not a generated-answer failure.",
            "Only the frozen qrels define relevance for this analysis.",
        ],
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
