from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
PROVENANCE_FIELDS = (
    "chunk_id",
    "source_uri",
    "document_version",
    "tenant_id",
    "groups",
    "shard_id",
    "index_version",
)


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def execute(dataset: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    if scenario["query_type"] != "text":
        return {
            "scenario_id": scenario["scenario_id"],
            "status": "unsupported",
            "partial": False,
            "failed_shards": [],
            "failed_components": [],
            "stale_index": False,
            "results": [],
        }

    tenant = scenario["tenant_id"]
    groups = set(scenario["groups"]) | {"all"}
    missing_shards = set(scenario.get("missing_shards", []))
    missing_components = list(scenario.get("missing_components", []))
    query_toks = tokens(scenario["query"])
    scored: list[tuple[float, str, dict[str, Any]]] = []

    for doc in dataset["documents"]:
        # Authorization is applied before scoring/candidate construction.
        if doc["tenant_id"] != tenant:
            continue
        if not groups.intersection(doc["groups"]):
            continue
        if doc["shard_id"] in missing_shards:
            continue
        overlap = len(query_toks & tokens(doc["text"]))
        if overlap:
            scored.append((float(overlap), doc["chunk_id"], doc))

    scored.sort(key=lambda row: (-row[0], row[1]))
    results = []
    for score, _, doc in scored[:3]:
        results.append(
            {field: doc[field] for field in PROVENANCE_FIELDS} | {"score": score}
        )

    stale = any(
        row["index_version"] < scenario["current_index_version"] for row in results
    )
    partial = bool(missing_shards or missing_components)
    status = "stale" if stale else ("partial" if partial else "ok")
    return {
        "scenario_id": scenario["scenario_id"],
        "status": status,
        "partial": partial,
        "failed_shards": sorted(missing_shards),
        "failed_components": sorted(missing_components),
        "stale_index": stale,
        "results": results,
    }


def evaluate(dataset: dict[str, Any]) -> dict[str, Any]:
    run_a = [execute(dataset, s) for s in dataset["scenarios"]]
    run_b = [execute(dataset, s) for s in dataset["scenarios"]]
    by_id = {s["scenario_id"]: s for s in dataset["scenarios"]}

    unauthorized = 0
    provenance_total = 0
    provenance_complete = 0
    status_correct = 0
    partial_correct = 0
    stale_correct = 0
    unsupported_correct = 0
    retrieval_expectations_correct = 0
    details = []

    for result in run_a:
        scenario = by_id[result["scenario_id"]]
        expected = scenario["expect"]
        returned = {r["chunk_id"] for r in result["results"]}
        forbidden = set(expected["forbidden_ids"])
        allowed = set(expected["allowed_ids"])
        leaked = sorted(returned & forbidden)
        unauthorized += len(leaked)
        for row in result["results"]:
            provenance_total += 1
            provenance_complete += int(
                all(
                    field in row and row[field] not in (None, "")
                    for field in PROVENANCE_FIELDS
                )
            )
        status_ok = result["status"] == expected["status"]
        partial_ok = result["partial"] == expected["partial"]
        stale_ok = result["stale_index"] == expected["stale_index"]
        retrieve_ok = allowed.issubset(returned) and not leaked
        unsupported_ok = True
        if scenario["query_type"] != "text":
            unsupported_ok = result["status"] == "unsupported" and not result["results"]
            unsupported_correct += int(unsupported_ok)
        status_correct += int(status_ok)
        partial_correct += int(partial_ok)
        stale_correct += int(stale_ok)
        retrieval_expectations_correct += int(retrieve_ok)
        details.append(
            {
                "scenario_id": result["scenario_id"],
                "returned_ids": sorted(returned),
                "leaked_forbidden_ids": leaked,
                "status_correct": status_ok,
                "partial_signal_correct": partial_ok,
                "stale_signal_correct": stale_ok,
                "retrieval_expectation_correct": retrieve_ok,
                "unsupported_behavior_correct": unsupported_ok,
            }
        )

    n = len(run_a)
    reproducible = canonical_hash(run_a) == canonical_hash(run_b)
    metrics = {
        "scenario_count": n,
        "unauthorized_return_count": unauthorized,
        "provenance_completeness": provenance_complete / provenance_total
        if provenance_total
        else 1.0,
        "status_accuracy": status_correct / n,
        "partial_result_signaling_accuracy": partial_correct / n,
        "stale_index_signaling_accuracy": stale_correct / n,
        "unsupported_query_accuracy": unsupported_correct
        / max(1, sum(s["query_type"] != "text" for s in dataset["scenarios"])),
        "retrieval_expectation_accuracy": retrieval_expectations_correct / n,
        "deterministic_reproducibility": reproducible,
        "run_hash": canonical_hash(run_a),
    }
    gates = {
        "unauthorized_return_count_eq_0": unauthorized == 0,
        "provenance_completeness_eq_1": metrics["provenance_completeness"] == 1.0,
        "partial_result_signaling_accuracy_eq_1": metrics[
            "partial_result_signaling_accuracy"
        ]
        == 1.0,
        "stale_index_signaling_accuracy_eq_1": metrics["stale_index_signaling_accuracy"]
        == 1.0,
        "unsupported_query_accuracy_eq_1": metrics["unsupported_query_accuracy"] == 1.0,
        "retrieval_expectation_accuracy_eq_1": metrics["retrieval_expectation_accuracy"]
        == 1.0,
        "deterministic_reproducibility_true": reproducible,
    }
    return {
        "schema_version": "atlasrag.safety-evaluation.v1",
        "scope": "fully synthetic, offline, deterministic contract harness",
        "metrics": metrics,
        "gates": {"passed": all(gates.values()), "checks": gates},
        "details": details,
        "results": run_a,
        "limitations": [
            "This harness validates explicit contract behavior over synthetic fixtures; it is not a penetration test.",
            "Authorization logic is intentionally small and deterministic; production identity, policy, and storage systems are not represented.",
            "Staleness and component failures are injected fixtures, not observed distributed-system incidents.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    report = evaluate(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["gates"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
