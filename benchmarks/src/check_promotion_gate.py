"""Compile frozen A/B evidence and apply AtlasRAG default-promotion gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from atlasrag.evaluation import evaluate_promotion

METRICS = ("recall@10", "mrr@10", "ndcg@10", "success@10")
FORBIDDEN_TEXT_KEYS = {"text", "query_text", "corpus_text", "document_text"}
CITATION_KEYS = {
    "external_id",
    "chunk_id",
    "document_id",
    "document_content_sha256",
    "content_sha256",
    "source_uri",
    "start_char",
    "end_char",
    "strategy_id",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="ascii") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def contains_forbidden_text_key(value: Any) -> bool:
    if isinstance(value, dict):
        if FORBIDDEN_TEXT_KEYS & set(value):
            return True
        return any(contains_forbidden_text_key(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_forbidden_text_key(item) for item in value)
    return False


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
    if not values:
        raise ValueError("bootstrap requires at least one value")
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
        "protocol_id": run["protocol_id"],
        "task_id": run["task_id"],
        "dataset": run["dataset"],
        "package": run["package"],
        "configuration": run["configuration"],
        "models": run["models"],
        "quality": run["quality"],
        "environment": run["environment"],
        "limitations": run["limitations"],
    }


def citation_complete(item: dict[str, Any]) -> bool:
    if not CITATION_KEYS.issubset(item):
        return False
    for key in (
        "external_id",
        "chunk_id",
        "document_id",
        "document_content_sha256",
        "content_sha256",
        "source_uri",
        "strategy_id",
    ):
        if not isinstance(item[key], str) or not item[key]:
            return False
    for key in ("document_content_sha256", "content_sha256"):
        value = item[key]
        if len(value) != 64:
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
    return (
        isinstance(item["start_char"], int)
        and isinstance(item["end_char"], int)
        and item["end_char"] > item["start_char"] >= 0
        and item["source_uri"].startswith("dataset://")
    )


def quality_analysis(
    rows: list[dict[str, Any]], *, task_id: str, seed: int, samples: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    task_offset = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)
    for metric_index, metric in enumerate(METRICS):
        deltas: list[float] = []
        for row in rows:
            baseline = metric_from_ids(
                list(row["baseline_top"]), list(row["relevant_ids"]), metric
            )
            candidate_ids = [str(item["external_id"]) for item in row["reranked_top"]]
            candidate = metric_from_ids(
                candidate_ids, list(row["relevant_ids"]), metric
            )
            deltas.append(candidate - baseline)
        lower, upper = bootstrap_interval(
            deltas,
            seed=seed + task_offset + metric_index,
            samples=samples,
        )
        output[metric] = {
            "mean_delta": statistics.fmean(deltas),
            "bootstrap_95_percent_interval": [lower, upper],
            "improved_queries": sum(delta > 0 for delta in deltas),
            "degraded_queries": sum(delta < 0 for delta in deltas),
            "unchanged_queries": sum(delta == 0 for delta in deltas),
            "interval_excludes_zero": lower > 0 or upper < 0,
        }
    return output


def process_task(
    *,
    run_a_path: Path,
    run_b_path: Path,
    rankings_a_path: Path,
    rankings_b_path: Path,
    summary_a_path: Path,
    summary_b_path: Path,
    seed: int,
    bootstrap_samples: int,
) -> tuple[str, dict[str, Any]]:
    run_a = load_json(run_a_path)
    run_b = load_json(run_b_path)
    rows_a = load_jsonl(rankings_a_path)
    rows_b = load_jsonl(rankings_b_path)
    task_id = str(run_a["task_id"])
    if run_b["task_id"] != task_id:
        raise ValueError("A/B task IDs differ")
    expected_count = int(run_a["dataset"]["query_count"])
    if len(rows_a) != len(rows_b) or len(rows_a) != expected_count:
        raise ValueError(f"unexpected query count for {task_id}")

    privacy_clean = not contains_forbidden_text_key(rows_a)
    citation_items = [
        item for row in rows_a for item in list(row.get("reranked_top", []))
    ]
    citation_count = len(citation_items)
    complete_count = sum(citation_complete(item) for item in citation_items)
    citation_completeness = complete_count / citation_count if citation_count else 0.0

    reranker_p95_values = [
        float(run_a["latency_ms"]["reranker_only"]["p95_ms"]),
        float(run_b["latency_ms"]["reranker_only"]["p95_ms"]),
    ]
    minimum_p95 = min(reranker_p95_values)
    ratio = max(reranker_p95_values) / minimum_p95 if minimum_p95 > 0 else math.inf
    controlled_host = all(
        bool(run["host_control"]["passed"])
        and bool(run["host_control"]["exclusive_benchmark_lock"])
        and bool(run["host_control"]["offline_model_loading"])
        and bool(run["host_control"]["one_benchmark_process"])
        for run in (run_a, run_b)
    )
    dispersion_acceptable = all(
        bool(run["latency_ms"]["dispersion_acceptable"]) for run in (run_a, run_b)
    )

    identity_equal = invariant_projection(run_a) == invariant_projection(run_b)
    quality_equal = run_a["quality"] == run_b["quality"]
    rankings_equal = rankings_a_path.read_bytes() == rankings_b_path.read_bytes()
    summary_equal = summary_a_path.read_bytes() == summary_b_path.read_bytes()

    evidence = {
        "candidate_depth": int(run_a["configuration"]["candidate_depth"]),
        "identity": {
            "protocol_id": run_a["protocol_id"],
            "dataset": run_a["dataset"],
            "package": run_a["package"],
            "configuration": run_a["configuration"],
            "models": run_a["models"],
        },
        "reproducibility": {
            "identity_equal": identity_equal,
            "rankings_byte_equal": rankings_equal,
            "quality_equal": quality_equal,
            "summary_byte_equal": summary_equal,
        },
        "privacy": {
            "payload_redistributed": bool(
                run_a["dataset"]["payload_redistributed"]
                or run_b["dataset"]["payload_redistributed"]
                or not privacy_clean
            ),
            "forbidden_text_keys_detected": not privacy_clean,
        },
        "citations": {
            "complete_count": complete_count,
            "result_count": citation_count,
            "completeness": citation_completeness,
        },
        "quality": quality_analysis(
            rows_a,
            task_id=task_id,
            seed=seed,
            samples=bootstrap_samples,
        ),
        "latency": {
            "controlled_host": controlled_host,
            "dispersion_acceptable": dispersion_acceptable,
            "reranker_p95_ms": reranker_p95_values,
            "reranker_p95_max_ms": max(reranker_p95_values),
            "reranker_p95_ratio": ratio,
            "scope": "controlled local reranker-component budget, not a production SLO",
        },
        "artifacts": {
            "run_a_sha256": sha256(run_a_path),
            "run_b_sha256": sha256(run_b_path),
            "rankings_a_sha256": sha256(rankings_a_path),
            "rankings_b_sha256": sha256(rankings_b_path),
            "summary_a_sha256": sha256(summary_a_path),
            "summary_b_sha256": sha256(summary_b_path),
        },
    }
    return task_id, evidence


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_json(args.gates)
    contracts_artifact = load_json(args.contracts)
    if contracts_artifact["protocol_id"] != policy["protocol_id"]:
        raise ValueError("contract artifact protocol ID does not match gates")

    task_inputs = (
        (
            args.scifact_run_a,
            args.scifact_run_b,
            args.scifact_rankings_a,
            args.scifact_rankings_b,
            args.scifact_summary_a,
            args.scifact_summary_b,
        ),
        (
            args.arguana_run_a,
            args.arguana_run_b,
            args.arguana_rankings_a,
            args.arguana_rankings_b,
            args.arguana_summary_a,
            args.arguana_summary_b,
        ),
    )
    tasks: dict[str, Any] = {}
    for paths in task_inputs:
        task_id, evidence = process_task(
            run_a_path=paths[0],
            run_b_path=paths[1],
            rankings_a_path=paths[2],
            rankings_b_path=paths[3],
            summary_a_path=paths[4],
            summary_b_path=paths[5],
            seed=int(policy["statistics"]["seed"]),
            bootstrap_samples=int(policy["statistics"]["bootstrap_samples"]),
        )
        if task_id in tasks:
            raise ValueError(f"duplicate task evidence: {task_id}")
        tasks[task_id] = evidence

    normalized = {
        "schema_version": "atlasrag.promotion-normalized-evidence.v1",
        "protocol_id": policy["protocol_id"],
        "contracts": contracts_artifact["contracts"],
        "tasks": tasks,
    }
    decision = evaluate_promotion(policy, normalized)
    output = {
        "schema_version": "atlasrag.promotion-gate-report.v1",
        "protocol_id": policy["protocol_id"],
        "gates_sha256": sha256(args.gates),
        "contracts_sha256": sha256(args.contracts),
        "evaluation_complete": True,
        "candidate_promoted": decision.enable_candidate_by_default,
        "decision": decision.to_dict(),
        "evidence": normalized,
        "failure_behavior": (
            "Non-promoted candidates exit nonzero unless --allow-retain-default is "
            "used solely to publish the failed-closed evidence report."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if not decision.enable_candidate_by_default and not args.allow_retain_default:
        raise SystemExit(2)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    for dataset in ("scifact", "arguana"):
        parser.add_argument(f"--{dataset}-run-a", type=Path, required=True)
        parser.add_argument(f"--{dataset}-run-b", type=Path, required=True)
        parser.add_argument(f"--{dataset}-rankings-a", type=Path, required=True)
        parser.add_argument(f"--{dataset}-rankings-b", type=Path, required=True)
        parser.add_argument(f"--{dataset}-summary-a", type=Path, required=True)
        parser.add_argument(f"--{dataset}-summary-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-retain-default", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
