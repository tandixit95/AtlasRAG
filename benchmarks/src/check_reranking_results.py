"""Validate reranking A/B reproducibility and compute paired analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

METRICS = ("recall@10", "mrr@10", "ndcg@10", "success@10")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def metric_from_ids(ranked: list[str], relevant_ids: list[str], metric: str) -> float:
    relevant = set(relevant_ids)
    hits = [1 if item in relevant else 0 for item in ranked[:10]]
    if metric == "recall@10":
        return sum(hits) / max(1, len(relevant))
    if metric == "mrr@10":
        return next((1.0 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    if metric == "ndcg@10":
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
        ideal_count = min(10, len(relevant))
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        return dcg / ideal if ideal else 0.0
    if metric == "success@10":
        return float(any(hits))
    raise ValueError(f"unsupported metric: {metric}")


def bootstrap_interval(
    values: list[float], *, seed: int, samples: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    count = len(values)
    means = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    means.sort()
    lower = means[int(samples * 0.025)]
    upper = means[min(samples - 1, int(samples * 0.975))]
    return lower, upper


def invariant_projection(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": run["schema_version"],
        "dataset": run["dataset"],
        "package": run["package"],
        "configuration": run["configuration"],
        "models": run["models"],
        "quality": run["quality"],
        "candidate_recall": run["candidate_recall"],
        "environment": run["environment"],
        "limitations": run["limitations"],
    }


def paired_analysis(
    rows: list[dict[str, Any]], *, depths: list[int], bootstrap_samples: int
) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    for depth in depths:
        depth_key = str(depth)
        metric_analysis: dict[str, Any] = {}
        for metric_index, metric in enumerate(METRICS):
            deltas: list[float] = []
            for row in rows:
                baseline = metric_from_ids(
                    row["hybrid_top"], row["relevant_ids"], metric
                )
                reranked = float(row["reranked"][depth_key]["metrics"][metric])
                deltas.append(reranked - baseline)
            lower, upper = bootstrap_interval(
                deltas,
                seed=20260802 + depth * 100 + metric_index,
                samples=bootstrap_samples,
            )
            metric_analysis[metric] = {
                "mean_delta": statistics.fmean(deltas),
                "bootstrap_95_percent_interval": [lower, upper],
                "improved_queries": sum(delta > 0 for delta in deltas),
                "degraded_queries": sum(delta < 0 for delta in deltas),
                "unchanged_queries": sum(delta == 0 for delta in deltas),
                "interval_excludes_zero": lower > 0 or upper < 0,
            }

        rescues = 0
        losses = 0
        for row in rows:
            relevant = set(row["relevant_ids"])
            baseline_success = bool(set(row["hybrid_top"][:10]) & relevant)
            reranked_success = bool(
                {item["external_id"] for item in row["reranked"][depth_key]["top"]}
                & relevant
            )
            rescues += int(not baseline_success and reranked_success)
            losses += int(baseline_success and not reranked_success)
        analysis[depth_key] = {
            "metrics": metric_analysis,
            "success_rescues": rescues,
            "success_losses": losses,
        }
    return analysis


def average_latency_p95(
    run_a: dict[str, Any], run_b: dict[str, Any], *, depth: int
) -> float:
    key = str(depth)
    values = [
        float(run_a["latency_ms"]["reranker_only"][key]["p95_ms"]),
        float(run_b["latency_ms"]["reranker_only"][key]["p95_ms"]),
    ]
    return statistics.fmean(values)


def latency_diagnostics(run: dict[str, Any]) -> dict[str, Any]:
    """Describe timing dispersion without treating the heuristic as a test."""

    series: dict[str, dict[str, float]] = {
        "hybrid_candidate_generation": run["latency_ms"]["hybrid_candidate_generation"]
    }
    for family in ("reranker_only", "measured_end_to_end"):
        for depth, values in run["latency_ms"][family].items():
            series[f"{family}.{depth}"] = values

    diagnostics: dict[str, Any] = {}
    for name, values in series.items():
        p50 = float(values["p50_ms"])
        p95 = float(values["p95_ms"])
        maximum = float(values["max_ms"])
        diagnostics[name] = {
            "p95_to_p50_ratio": p95 / p50 if p50 else None,
            "max_to_p50_ratio": maximum / p50 if p50 else None,
            "high_dispersion_heuristic": (
                p50 == 0.0 or p95 > 3.0 * p50 or maximum > 10.0 * p50
            ),
        }
    return diagnostics


def dominated_depths(
    run_a: dict[str, Any], run_b: dict[str, Any], depths: list[int]
) -> dict[str, int]:
    dominated: dict[str, int] = {}
    for candidate in depths:
        candidate_quality = run_a["quality"][f"reranked_{candidate}"]
        candidate_latency = average_latency_p95(run_a, run_b, depth=candidate)
        for alternative in depths:
            if alternative == candidate:
                continue
            alternative_quality = run_a["quality"][f"reranked_{alternative}"]
            alternative_latency = average_latency_p95(run_a, run_b, depth=alternative)
            quality_not_worse = all(
                alternative_quality[metric] >= candidate_quality[metric]
                for metric in METRICS
            )
            strictly_better = (
                any(
                    alternative_quality[metric] > candidate_quality[metric]
                    for metric in METRICS
                )
                or alternative_latency < candidate_latency
            )
            if (
                quality_not_worse
                and alternative_latency <= candidate_latency
                and strictly_better
            ):
                dominated[str(candidate)] = alternative
                break
    return dominated


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_a = load_json(args.run_a)
    run_b = load_json(args.run_b)
    rows_a = load_jsonl(args.rankings_a)
    rows_b = load_jsonl(args.rankings_b)
    depths = [int(value) for value in run_a["configuration"]["candidate_depths"]]

    checks = {
        "run_invariants_equal": invariant_projection(run_a)
        == invariant_projection(run_b),
        "quality_equal": run_a["quality"] == run_b["quality"],
        "candidate_recall_equal": run_a["candidate_recall"]
        == run_b["candidate_recall"],
        "raw_rankings_byte_equal": args.rankings_a.read_bytes()
        == args.rankings_b.read_bytes(),
        "summary_csv_byte_equal": args.summary_a.read_bytes()
        == args.summary_b.read_bytes(),
        "query_count_equal": len(rows_a)
        == len(rows_b)
        == run_a["dataset"]["query_count"],
        "latency_query_ids_equal": [
            item["query_id"] for item in run_a["latency_ms"]["samples"]
        ]
        == [item["query_id"] for item in run_b["latency_ms"]["samples"]],
    }
    analysis = paired_analysis(
        rows_a,
        depths=depths,
        bootstrap_samples=args.bootstrap_samples,
    )
    latency = {
        "run_a": latency_diagnostics(run_a),
        "run_b": latency_diagnostics(run_b),
    }
    latency_high_dispersion = any(
        item["high_dispersion_heuristic"]
        for run in latency.values()
        for item in run.values()
    )
    dominated = dominated_depths(run_a, run_b, depths)
    all_intervals_include_zero = all(
        not value["interval_excludes_zero"]
        for depth in analysis.values()
        for value in depth["metrics"].values()
    )

    output = {
        "schema_version": "atlasrag.reranking-reproducibility.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "artifacts": {
            "run_a_sha256": sha256(args.run_a),
            "run_b_sha256": sha256(args.run_b),
            "rankings_a_sha256": sha256(args.rankings_a),
            "rankings_b_sha256": sha256(args.rankings_b),
            "summary_a_sha256": sha256(args.summary_a),
            "summary_b_sha256": sha256(args.summary_b),
        },
        "paired_analysis": analysis,
        "latency_diagnostics": {
            "post_hoc_heuristic": (
                "High dispersion when p95 exceeds 3x p50 or max exceeds 10x p50. "
                "This diagnostic was added after inspecting host contention and is "
                "not a preregistered statistical test."
            ),
            "high_dispersion_detected": latency_high_dispersion,
            "runs": latency,
        },
        "dominated_candidate_depths": dominated,
        "recommendation": {
            "enable_by_default": False,
            "efficient_point_estimate_depth": 10,
            "highest_quality_point_estimate_depth": 50,
            "rejected_depths": dominated,
            "reason": (
                "A/B rankings reproduce exactly, but every paired 95% bootstrap "
                "interval includes zero; evidence is limited to one dataset, one "
                "dense model, one reranker, and one host."
            ),
            "all_quality_delta_intervals_include_zero": all_intervals_include_zero,
            "publish_latency_as_stable_component_claim": not latency_high_dispersion,
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if not output["passed"]:
        raise SystemExit("reranking reproducibility checks failed")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--rankings-a", type=Path, required=True)
    parser.add_argument("--rankings-b", type=Path, required=True)
    parser.add_argument("--summary-a", type=Path, required=True)
    parser.add_argument("--summary-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
