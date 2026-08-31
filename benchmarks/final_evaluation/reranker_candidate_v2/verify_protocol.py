"""Fail-closed validation for the frozen reranker final-evaluation v2 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
PROTOCOL = ROOT / "PROTOCOL.json"
GATES = ROOT / "GATES.json"
PROFILE = REPO_ROOT / "benchmarks/development/reranker_candidate_v1/PROFILE.json"

EXPECTED_PROFILE_SHA256 = (
    "312072db352d7ef8029a40f74142338e0437f6957888528df47acc3eca13a22b"
)
EXPECTED_WORKLOAD_SHA256 = (
    "55047067d68ca04733954eb164eb90b99ef1181fff5bcb05081b139bcc80e482"
)
EXPECTED_PROTOCOL_ID = "reranked-depth10-batch32-final-evaluation-20260831"
EXPECTED_CANDIDATE = {"candidate_depth": 10, "batch_size": 32}
EXPECTED_EXECUTION = {
    "runner": "benchmarks/src/run_promotion_benchmark.py",
    "checker": "benchmarks/src/check_promotion_gate.py",
    "candidate_depth": 10,
    "top_k": 10,
    "hybrid_component_k": 100,
    "rrf_k": 60,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "reranker_batch_size": 32,
    "device": "cuda",
    "latency_sample": 25,
    "max_load_ratio": 0.5,
    "seed": 20260802,
    "independent_runs": ["a", "b"],
    "offline_model_loading_required": True,
    "exclusive_benchmark_lock_required": True,
    "source_commit_rule": (
        "evaluate a wheel built from the remote-verified commit containing this "
        "frozen protocol"
    ),
}
EXPECTED_STATISTICS = {
    "bootstrap_samples": 10000,
    "confidence_level": 0.95,
    "seed": 20260802,
    "primary_metric": "mrr@10",
    "secondary_metric": "ndcg@10",
    "recall_metric": "recall@10",
}
EXPECTED_TASKS = [
    {
        "task_id": "scifact-test-300",
        "dataset": "scifact",
        "query_count": 300,
        "selection_rule": "full sorted positive-qrel test set",
    },
    {
        "task_id": "arguana-contrast-200",
        "dataset": "arguana",
        "query_count": 200,
        "selection_rule": (
            "SHA256(atlasrag-arguana-contrast-20260731: + query_id), then "
            "query_id; take 200"
        ),
        "identical_query_document_ids_excluded": True,
    },
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _candidate_from_profile(profile: dict[str, Any]) -> dict[str, int] | None:
    candidates = profile.get("nominated_candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        return None
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return None
    try:
        return {
            "candidate_depth": int(candidate["candidate_depth"]),
            "batch_size": int(candidate["batch_size"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def validate(
    protocol: dict[str, Any],
    gates: dict[str, Any],
    profile: dict[str, Any],
    *,
    profile_sha256: str,
) -> list[str]:
    errors: list[str] = []
    if (
        protocol.get("schema_version")
        != "atlasrag.reranker-final-evaluation-protocol.v2"
    ):
        errors.append("protocol schema_version mismatch")
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        errors.append("protocol_id mismatch")
    if protocol.get("status") != "frozen_unexecuted":
        errors.append("protocol must remain frozen_unexecuted before outcomes")
    if profile_sha256 != EXPECTED_PROFILE_SHA256:
        errors.append("development profile SHA-256 mismatch")
    if profile.get("schema_version") != "atlasrag.reranker-development-profile.v1":
        errors.append("development profile schema mismatch")
    if profile.get("status") != "development_profile_complete":
        errors.append("development profile is not complete")
    if profile.get("workload_sha256") != EXPECTED_WORKLOAD_SHA256:
        errors.append("development workload SHA-256 mismatch")
    if _candidate_from_profile(profile) != EXPECTED_CANDIDATE:
        errors.append("development profile nomination differs from frozen candidate")

    dev = protocol.get("development_profile")
    if not isinstance(dev, dict):
        errors.append("development_profile must be an object")
    else:
        if dev.get("sha256") != EXPECTED_PROFILE_SHA256:
            errors.append("protocol development profile SHA-256 mismatch")
        nominated = dev.get("nominated_candidate")
        if (
            not isinstance(nominated, dict)
            or {
                "candidate_depth": nominated.get("candidate_depth"),
                "batch_size": nominated.get("batch_size"),
            }
            != EXPECTED_CANDIDATE
        ):
            errors.append("protocol nominated candidate mismatch")

    candidate = protocol.get("candidate")
    if not isinstance(candidate, dict):
        errors.append("candidate must be an object")
    else:
        if candidate.get("candidate_depth") != 10:
            errors.append("candidate depth must remain 10")
        if candidate.get("reranker_batch_size") != 32:
            errors.append("candidate reranker batch size must remain 32")
        if candidate.get("default_enabled_before_evaluation") is not False:
            errors.append("candidate must not be default before evaluation")

    if protocol.get("required_tasks") != EXPECTED_TASKS:
        errors.append("required final task shapes differ from the frozen definition")
    if protocol.get("execution") != EXPECTED_EXECUTION:
        errors.append("execution configuration differs from the frozen definition")
    if protocol.get("statistics") != EXPECTED_STATISTICS:
        errors.append("statistics differ from the frozen definition")

    handling = protocol.get("outcome_handling")
    expected_handling = {
        "post_outcome_candidate_tuning_allowed": False,
        "candidate_replacement_after_outcomes_allowed": False,
        "promotion_only_if_all_gates_pass": True,
        "retain_current_default_otherwise": True,
    }
    if handling != expected_handling:
        errors.append("outcome handling differs from the frozen definition")
    claim = protocol.get("claim_boundary")
    expected_claim = {
        "production_latency_claim_allowed": False,
        "throughput_claim_allowed": False,
        "final_quality_claim_allowed_before_execution": False,
        "default_promotion_allowed_before_execution": False,
    }
    if claim != expected_claim:
        errors.append("claim boundary differs from the frozen definition")

    if gates.get("schema_version") != "atlasrag.promotion-gates.v2":
        errors.append("gate schema_version mismatch")
    if gates.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        errors.append("gate protocol_id mismatch")
    if gates.get("status") != "frozen-before-outcomes":
        errors.append("gates must remain frozen-before-outcomes")
    gate_candidate = gates.get("candidate")
    if (
        not isinstance(gate_candidate, dict)
        or gate_candidate.get("reranker_batch_size") != 32
    ):
        errors.append("gate candidate batch size must remain 32")
    vetoes = gates.get("per_task_vetoes")
    expected_batch_veto = {
        "id": "reranker_batch_size_frozen",
        "path": ["identity", "configuration", "reranker_batch_size"],
        "operator": "eq",
        "threshold": 32,
    }
    if not isinstance(vetoes, list) or expected_batch_veto not in vetoes:
        errors.append("frozen reranker batch-size veto is missing")
    p95_veto = None
    if isinstance(vetoes, list):
        p95_veto = next(
            (item for item in vetoes if item.get("id") == "reranker_p95_budget_ms"),
            None,
        )
    if not isinstance(p95_veto, dict) or p95_veto.get("threshold") != 75.0:
        errors.append("final reranker p95 budget must remain 75 ms")
    stats = gates.get("statistics")
    if stats != EXPECTED_STATISTICS:
        errors.append("gate statistics differ from the frozen definition")
    return errors


def main() -> int:
    profile_bytes = PROFILE.read_bytes()
    errors = validate(
        json.loads(PROTOCOL.read_text()),
        json.loads(GATES.read_text()),
        json.loads(profile_bytes),
        profile_sha256=_sha256(profile_bytes),
    )
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(
        "PASS: reranker final-evaluation v2 protocol is frozen before outcomes "
        "at depth 10 / batch 32"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
