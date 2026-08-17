"""Fail-closed validation for the reranker development-only protocol."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "PROTOCOL.json"
EXPECTED_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
EXPECTED_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
FROZEN_FINAL_DATASETS = {"scifact", "arguana"}


def validate(protocol: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if protocol.get("schema_version") != "atlasrag.reranker-development-protocol.v1":
        errors.append("schema_version mismatch")
    if protocol.get("status") != "development_only_unexecuted":
        errors.append("protocol must remain explicitly unexecuted before profiling")
    if protocol.get("final_evaluation_data_prohibited") is not True:
        errors.append("final evaluation data must be prohibited")
    prohibited = {
        str(v).casefold() for v in protocol.get("prohibited_dataset_names", [])
    }  # type: ignore[arg-type]
    if not FROZEN_FINAL_DATASETS.issubset(prohibited):
        errors.append("frozen final datasets must remain explicitly prohibited")
    source = protocol.get("input_source")
    if not isinstance(source, dict):
        errors.append("input_source must be an object")
    else:
        if source.get("type") != "deterministic_synthetic_templates":
            errors.append(
                "development inputs must be deterministic synthetic templates"
            )
        if source.get("payload_from_final_evaluation_sets") is not False:
            errors.append("final evaluation payload use must be false")
        if int(source.get("query_count", 0)) <= 0:
            errors.append("query_count must be positive")
    reranker = protocol.get("reranker")
    if not isinstance(reranker, dict):
        errors.append("reranker must be an object")
    else:
        if reranker.get("name") != EXPECTED_RERANKER:
            errors.append("reranker identity mismatch")
        if reranker.get("revision") != EXPECTED_REVISION:
            errors.append("reranker revision mismatch")
        if reranker.get("offline_local_snapshot_required") is not True:
            errors.append("offline local snapshot must be required")
    matrix = protocol.get("matrix")
    if not isinstance(matrix, dict):
        errors.append("matrix must be an object")
    else:
        if matrix.get("candidate_depths") != [10, 20, 50]:
            errors.append("candidate depths must remain [10, 20, 50]")
        batches = matrix.get("batch_sizes")
        if (
            not isinstance(batches, list)
            or not batches
            or any(int(x) <= 0 for x in batches)
        ):
            errors.append("batch sizes must be positive")
        if int(matrix.get("warmup_iterations", 0)) < 1:
            errors.append("at least one warmup iteration is required")
        if int(matrix.get("measured_iterations", 0)) < 10:
            errors.append("at least ten measured iterations are required")
    controls = protocol.get("host_controls")
    required = [
        "exclusive_benchmark_lock_required",
        "single_profiler_process",
        "network_disabled_for_model_load",
        "accelerator_synchronization_required_when_available",
        "raw_timing_samples_required",
        "environment_metadata_required",
    ]
    if not isinstance(controls, dict) or any(
        controls.get(k) is not True for k in required
    ):
        errors.append("all host controls must fail closed")
    gates = protocol.get("candidate_gates")
    if not isinstance(gates, dict):
        errors.append("candidate_gates must be an object")
    else:
        for k in [
            "finite_scores_required",
            "score_count_match_required",
            "repeat_order_determinism_required",
        ]:
            if gates.get(k) is not True:
                errors.append(f"{k} must be required")
        if float(gates.get("batch_size_p95_over_best_same_depth_max_ratio", 99)) > 1.10:
            errors.append("batch-size p95 selection tolerance may not exceed 1.10")
        if int(gates.get("maximum_nominated_candidates", 0)) != 1:
            errors.append("at most one candidate may be nominated")
    boundary = protocol.get("claim_boundary")
    if not isinstance(boundary, dict):
        errors.append("claim_boundary must be an object")
    else:
        for k in [
            "production_latency_claim_allowed",
            "throughput_claim_allowed",
            "quality_claim_allowed",
            "default_promotion_allowed",
        ]:
            if boundary.get(k) is not False:
                errors.append(
                    "development protocol may not authorize public performance "
                    "or promotion claims"
                )
        if boundary.get("final_evaluation_required_after_candidate_freeze") is not True:
            errors.append(
                "a new final evaluation must be required after candidate freeze"
            )
    return errors


def main() -> int:
    errors = validate(json.loads(PROTOCOL.read_text(encoding="utf-8")))
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(
        "PASS: reranker development protocol is isolated from frozen final "
        "evaluation data"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
