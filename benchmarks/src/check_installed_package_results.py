from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METHODS = ("bm25", "exact_dense", "hybrid_rrf")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_raw(result_path: Path, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_path = result_path.parent / payload["artifacts"]["raw_queries"]
    rows: dict[str, dict[str, Any]] = {}
    with raw_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["query_id"]] = row
    return rows


def quality_delta(
    installed: dict[str, Any], neutral: dict[str, Any]
) -> dict[str, dict[str, float]]:
    return {
        method: {
            metric: installed["quality"][method][metric]
            - neutral["quality"][method][metric]
            for metric in installed["quality"][method]
        }
        for method in METHODS
    }


def compare_rankings(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    query_ids = sorted(set(left) | set(right))
    mismatches: list[dict[str, Any]] = []
    equal_queries = 0
    equal_by_method = {method: 0 for method in METHODS}
    for query_id in query_ids:
        query_equal = query_id in left and query_id in right
        method_differences: dict[str, Any] = {}
        if query_equal:
            for method in METHODS:
                left_ids = left[query_id]["top"][method]
                right_ids = right[query_id]["top"][method]
                if left_ids == right_ids:
                    equal_by_method[method] += 1
                else:
                    query_equal = False
                    method_differences[method] = {
                        "left": left_ids,
                        "right": right_ids,
                    }
        if query_equal:
            equal_queries += 1
        elif len(mismatches) < 10:
            mismatches.append(
                {
                    "query_id": query_id,
                    "method_differences": method_differences,
                    "missing_left": query_id not in left,
                    "missing_right": query_id not in right,
                }
            )
    return {
        "queries": len(query_ids),
        "equal_queries": equal_queries,
        "equal_by_method": equal_by_method,
        "all_equal": equal_queries == len(query_ids),
        "mismatch_examples": mismatches,
    }


def compare_dataset(
    *,
    name: str,
    a_path: Path,
    b_path: Path,
    neutral_path: Path,
) -> dict[str, Any]:
    a = load(a_path)
    b = load(b_path)
    neutral = load(neutral_path)
    raw_a = load_raw(a_path, a)
    raw_b = load_raw(b_path, b)
    raw_neutral = load_raw(neutral_path, neutral)
    package_rankings = compare_rankings(raw_a, raw_b)
    neutral_rankings = compare_rankings(raw_a, raw_neutral)
    package_identity_equal = all(
        a["atlasrag"][field] == b["atlasrag"][field]
        for field in ("version", "git_commit", "wheel_sha256")
    )
    dataset_hashes_equal = all(
        a["dataset"][field] == b["dataset"][field]
        for field in ("corpus_sha256", "queries_sha256", "qrels_sha256")
    )
    model_equal = a["model"]["revision"] == b["model"]["revision"]
    quality_equal = a["quality"] == b["quality"]
    passed = (
        package_identity_equal
        and dataset_hashes_equal
        and model_equal
        and quality_equal
        and package_rankings["all_equal"]
    )
    deltas = quality_delta(a, neutral)
    neutral_quality_equal = all(
        value == 0.0 for method in deltas.values() for value in method.values()
    )
    return {
        "dataset": name,
        "passed": passed,
        "package_identity_equal": package_identity_equal,
        "dataset_hashes_equal": dataset_hashes_equal,
        "model_revision_equal": model_equal,
        "quality_equal_between_package_runs": quality_equal,
        "package_rerun_rankings": package_rankings,
        "neutral_harness_comparison": {
            "quality_delta_installed_minus_neutral": deltas,
            "quality_equal": neutral_quality_equal,
            "rankings": neutral_rankings,
            "note": (
                "Neutral equivalence is informative, not a release gate. Equal or nearly "
                "equal scores can order differently because the installed package preserves "
                "its deterministic chunk-ID tie breaker and uses a different numeric path."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scifact-a", type=Path, required=True)
    parser.add_argument("--scifact-b", type=Path, required=True)
    parser.add_argument("--scifact-neutral", type=Path, required=True)
    parser.add_argument("--arguana-a", type=Path, required=True)
    parser.add_argument("--arguana-b", type=Path, required=True)
    parser.add_argument("--arguana-neutral", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = [
        compare_dataset(
            name="scifact",
            a_path=args.scifact_a,
            b_path=args.scifact_b,
            neutral_path=args.scifact_neutral,
        ),
        compare_dataset(
            name="arguana-contrast-200",
            a_path=args.arguana_a,
            b_path=args.arguana_b,
            neutral_path=args.arguana_neutral,
        ),
    ]
    payload = {
        "schema_version": "atlasrag.installed-package-reproducibility.v1",
        "passed": all(item["passed"] for item in datasets),
        "datasets": datasets,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
