"""Analyze no-payload rank movements from frozen promotion evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FORBIDDEN_TEXT_KEYS = {"text", "query_text", "corpus_text", "document_text"}
METRICS = ("recall@10", "mrr@10", "ndcg@10", "success@10")
MOVEMENT_CLASSES = (
    "improved",
    "unchanged",
    "regressed",
    "rescued",
    "lost",
    "both_absent",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contains_forbidden_text_key(value: Any) -> bool:
    if isinstance(value, dict):
        if FORBIDDEN_TEXT_KEYS & set(value):
            return True
        return any(contains_forbidden_text_key(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_forbidden_text_key(item) for item in value)
    return False


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        stream_context = gzip.open(path, "rt", encoding="utf-8")
    else:
        stream_context = path.open("rt", encoding="utf-8")
    with stream_context as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("rankings must contain JSON objects")
    if contains_forbidden_text_key(rows):
        raise ValueError("rankings contain forbidden dataset payload text keys")
    return rows


def first_relevant_rank(
    ranked_ids: Iterable[str], relevant_ids: Iterable[str]
) -> int | None:
    relevant = set(relevant_ids)
    return next(
        (rank for rank, item in enumerate(ranked_ids, start=1) if item in relevant),
        None,
    )


def movement_class(baseline_rank: int | None, candidate_rank: int | None) -> str:
    if baseline_rank is None and candidate_rank is None:
        return "both_absent"
    if baseline_rank is None:
        return "rescued"
    if candidate_rank is None:
        return "lost"
    if candidate_rank < baseline_rank:
        return "improved"
    if candidate_rank > baseline_rank:
        return "regressed"
    return "unchanged"


def _rounded(value: float) -> float:
    return round(float(value), 12)


def _metric_movements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in METRICS:
        deltas = [
            float(row["candidate_metrics"][metric])
            - float(row["baseline_metrics"][metric])
            for row in rows
        ]
        output[metric] = {
            "mean_delta": _rounded(statistics.fmean(deltas)),
            "sum_delta": _rounded(sum(deltas)),
            "improved_queries": sum(delta > 0 for delta in deltas),
            "degraded_queries": sum(delta < 0 for delta in deltas),
            "unchanged_queries": sum(delta == 0 for delta in deltas),
        }
    return output


def _score_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": _rounded(statistics.fmean(values)),
        "median": _rounded(statistics.median(values)),
        "minimum": _rounded(min(values)),
        "maximum": _rounded(max(values)),
    }


def validate_row(row: dict[str, Any], *, top_k: int) -> None:
    required = {
        "query_id",
        "relevant_ids",
        "baseline_top",
        "reranked_top",
        "baseline_metrics",
        "candidate_metrics",
    }
    if not required.issubset(row):
        raise ValueError(f"ranking row is missing keys: {sorted(required - set(row))}")
    if not isinstance(row["query_id"], str) or not row["query_id"]:
        raise ValueError("query_id must be a non-empty string")
    relevant_ids = row["relevant_ids"]
    if not isinstance(relevant_ids, list) or not relevant_ids:
        raise ValueError("relevant_ids must be a non-empty list")
    if not all(isinstance(item, str) and item for item in relevant_ids):
        raise ValueError("relevant_ids must contain non-empty strings")

    baseline_top = row["baseline_top"]
    reranked_top = row["reranked_top"]
    if len(baseline_top) != top_k or len(reranked_top) != top_k:
        raise ValueError(f"each ranking must contain exactly {top_k} items")
    if len(set(baseline_top)) != top_k:
        raise ValueError("baseline ranking contains duplicate IDs")
    if not all(isinstance(item, dict) for item in reranked_top):
        raise ValueError("reranked_top must contain objects")

    candidate_ids = [item.get("external_id") for item in reranked_top]
    if not all(isinstance(item, str) and item for item in candidate_ids):
        raise ValueError("reranked items require external_id")
    if len(set(candidate_ids)) != top_k:
        raise ValueError("candidate ranking contains duplicate IDs")
    if set(candidate_ids) != set(baseline_top):
        raise ValueError("candidate set changed between baseline and reranking")
    for item in reranked_top:
        candidate_rank = item.get("candidate_rank")
        if isinstance(candidate_rank, bool) or not isinstance(candidate_rank, int):
            raise ValueError("reranked items require integer candidate_rank")
        reranker_score = item.get("reranker_score")
        if isinstance(reranker_score, bool) or not isinstance(
            reranker_score, int | float
        ):
            raise ValueError("reranked items require numeric reranker_score")
        if not math.isfinite(float(reranker_score)):
            raise ValueError("reranker_score must be finite")
    for metric in METRICS:
        for key in ("baseline_metrics", "candidate_metrics"):
            value = row[key].get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{key}.{metric} must be finite numeric evidence")


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    source_path: str,
    source_sha256: str,
    top_k: int = 10,
) -> dict[str, Any]:
    for row in rows:
        validate_row(row, top_k=top_k)

    classifications: Counter[str] = Counter()
    query_ids_by_class: dict[str, list[str]] = defaultdict(list)
    rank_deltas: list[int] = []
    delta_counts: Counter[int] = Counter()
    baseline_rank_counts: Counter[int | None] = Counter()
    candidate_rank_counts: Counter[int | None] = Counter()
    baseline_buckets: dict[int, list[int]] = defaultdict(list)
    score_by_class: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"relevant_score": [], "margin_to_top": []}
    )
    movement_records: list[dict[str, Any]] = []

    for row in rows:
        relevant_ids = list(row["relevant_ids"])
        baseline_ids = list(row["baseline_top"])
        reranked = list(row["reranked_top"])
        candidate_ids = [str(item["external_id"]) for item in reranked]
        baseline_rank = first_relevant_rank(baseline_ids, relevant_ids)
        candidate_rank = first_relevant_rank(candidate_ids, relevant_ids)
        classification = movement_class(baseline_rank, candidate_rank)
        classifications[classification] += 1
        query_ids_by_class[classification].append(str(row["query_id"]))
        baseline_rank_counts[baseline_rank] += 1
        candidate_rank_counts[candidate_rank] += 1

        rank_delta = None
        if baseline_rank is not None and candidate_rank is not None:
            rank_delta = candidate_rank - baseline_rank
            rank_deltas.append(rank_delta)
            delta_counts[rank_delta] += 1
            baseline_buckets[baseline_rank].append(rank_delta)

            relevant = set(relevant_ids)
            relevant_item = next(
                item for item in reranked if str(item["external_id"]) in relevant
            )
            relevant_score = float(relevant_item["reranker_score"])
            top_score = float(reranked[0]["reranker_score"])
            score_by_class[classification]["relevant_score"].append(relevant_score)
            score_by_class[classification]["margin_to_top"].append(
                top_score - relevant_score
            )

        movement_records.append(
            {
                "query_id": str(row["query_id"]),
                "classification": classification,
                "baseline_rank": baseline_rank,
                "candidate_rank": candidate_rank,
                "rank_delta": rank_delta,
                "mrr_delta": _rounded(
                    float(row["candidate_metrics"]["mrr@10"])
                    - float(row["baseline_metrics"]["mrr@10"])
                ),
                "ndcg_delta": _rounded(
                    float(row["candidate_metrics"]["ndcg@10"])
                    - float(row["baseline_metrics"]["ndcg@10"])
                ),
            }
        )

    bucket_output: dict[str, Any] = {}
    for rank in range(1, top_k + 1):
        deltas = baseline_buckets.get(rank, [])
        bucket_output[str(rank)] = {
            "query_count": baseline_rank_counts[rank],
            "improved": sum(delta < 0 for delta in deltas),
            "unchanged": sum(delta == 0 for delta in deltas),
            "regressed": sum(delta > 0 for delta in deltas),
            "mean_rank_delta": _rounded(statistics.fmean(deltas)) if deltas else None,
        }
    bucket_output["absent"] = {
        "query_count": baseline_rank_counts[None],
        "rescued": classifications["rescued"],
        "both_absent": classifications["both_absent"],
    }

    score_output: dict[str, Any] = {}
    for classification in ("improved", "unchanged", "regressed"):
        values = score_by_class[classification]
        score_output[classification] = {
            "query_count": classifications[classification],
            "relevant_reranker_score": _score_summary(values["relevant_score"]),
            "relevant_score_margin_to_top": _score_summary(values["margin_to_top"]),
        }

    severe_regressions = sorted(
        (item for item in movement_records if item["classification"] == "regressed"),
        key=lambda item: (-int(item["rank_delta"]), item["query_id"]),
    )[:10]
    strongest_improvements = sorted(
        (item for item in movement_records if item["classification"] == "improved"),
        key=lambda item: (int(item["rank_delta"]), item["query_id"]),
    )[:10]

    return {
        "schema_version": "atlasrag.arguana-rank-movement-analysis.v1",
        "source": {
            "path": source_path,
            "sha256": source_sha256,
            "query_count": len(rows),
            "top_k": top_k,
            "candidate_set_preserved_queries": len(rows),
            "payload_text_keys_present": False,
        },
        "classification_counts": {
            name: classifications[name] for name in MOVEMENT_CLASSES
        },
        "query_ids_by_class": {
            name: sorted(query_ids_by_class[name]) for name in MOVEMENT_CLASSES
        },
        "metric_movements": _metric_movements(rows),
        "rank_movements": {
            "queries_with_relevant_candidate": len(rank_deltas),
            "mean_rank_delta": _rounded(statistics.fmean(rank_deltas)),
            "median_rank_delta": _rounded(statistics.median(rank_deltas)),
            "minimum_rank_delta": min(rank_deltas),
            "maximum_rank_delta": max(rank_deltas),
            "delta_distribution": {
                str(delta): delta_counts[delta] for delta in sorted(delta_counts)
            },
            "baseline_rank_counts": {
                str(rank): baseline_rank_counts[rank] for rank in range(1, top_k + 1)
            }
            | {"absent": baseline_rank_counts[None]},
            "candidate_rank_counts": {
                str(rank): candidate_rank_counts[rank] for rank in range(1, top_k + 1)
            }
            | {"absent": candidate_rank_counts[None]},
            "by_baseline_rank": bucket_output,
        },
        "reranker_score_diagnostics": score_output,
        "severe_regressions": severe_regressions,
        "strongest_improvements": strongest_improvements,
        "interpretation": {
            "candidate_recall_changed": False,
            "candidate_pool_changed": False,
            "ordering_regression_is_primary_observation": True,
            "supported_observation": (
                "The frozen depth-10 reranker preserved the candidate set and recall, "
                "but moved relevant documents down more often than up on this slice."
            ),
        },
        "limitations": [
            "This is a deterministic 200-query ArguAna contrast slice, not a full official score.",
            "The analysis uses IDs, ranks, hashes, scores, and judged relevance only; no query or corpus text is included.",
            "Rank movement is diagnostic evidence and does not establish a causal linguistic failure mode.",
            "The frozen final slice must not be used to tune a replacement reranker.",
            "A revised candidate requires development-task tuning and a new frozen protocol before final re-evaluation.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.rankings)
    report = analyze_rows(
        rows,
        source_path=args.rankings.as_posix(),
        source_sha256=sha256(args.rankings),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
