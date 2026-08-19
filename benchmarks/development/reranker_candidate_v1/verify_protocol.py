"""Fail-closed validation for the frozen reranker development protocol."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "PROTOCOL.json"

EXPECTED_INPUT_SOURCE = {
    "type": "deterministic_synthetic_templates",
    "seed": 20260817,
    "query_count": 32,
    "document_template_version": "synthetic-retrieval-component-v1",
    "payload_from_final_evaluation_sets": False,
}
EXPECTED_RERANKER = {
    "name": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "revision": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
    "license": "Apache-2.0",
    "offline_local_snapshot_required": True,
}
EXPECTED_MATRIX = {
    "candidate_depths": [10, 20, 50],
    "batch_sizes": [16, 32, 64],
    "warmup_iterations": 5,
    "measured_iterations": 30,
}
EXPECTED_HOST_CONTROLS = {
    "exclusive_benchmark_lock_required": True,
    "single_profiler_process": True,
    "network_disabled_for_model_load": True,
    "accelerator_synchronization_required_when_available": True,
    "raw_timing_samples_required": True,
    "environment_metadata_required": True,
}
EXPECTED_CANDIDATE_GATES = {
    "finite_scores_required": True,
    "score_count_match_required": True,
    "repeat_order_determinism_required": True,
    "batch_size_p95_over_best_same_depth_max_ratio": 1.1,
    "peak_accelerator_memory_mib_max": 6144,
    "default_candidate_depth": 10,
    "maximum_nominated_candidates": 1,
}
EXPECTED_CLAIM_BOUNDARY = {
    "production_latency_claim_allowed": False,
    "throughput_claim_allowed": False,
    "quality_claim_allowed": False,
    "default_promotion_allowed": False,
    "final_evaluation_required_after_candidate_freeze": True,
}
FROZEN_FINAL_DATASETS = {"scifact", "arguana"}


def _require_exact_object(
    protocol: dict[str, object],
    key: str,
    expected: dict[str, object],
    errors: list[str],
) -> None:
    value = protocol.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be an object")
    elif value != expected:
        errors.append(f"{key} differs from the frozen v1 definition")


def validate(protocol: dict[str, object]) -> list[str]:
    """Return every violation of the exact frozen development protocol."""

    errors: list[str] = []
    expected_top_level = {
        "schema_version",
        "status",
        "purpose",
        "final_evaluation_data_prohibited",
        "prohibited_dataset_names",
        "input_source",
        "reranker",
        "matrix",
        "host_controls",
        "candidate_gates",
        "claim_boundary",
    }
    if set(protocol) != expected_top_level:
        errors.append("top-level fields differ from the frozen v1 definition")
    if protocol.get("schema_version") != "atlasrag.reranker-development-protocol.v1":
        errors.append("schema_version mismatch")
    if protocol.get("status") != "development_only_unexecuted":
        errors.append("protocol must remain explicitly unexecuted before profiling")
    if protocol.get("purpose") != (
        "select at most one reranker configuration using synthetic development "
        "evidence before any new final evaluation freeze"
    ):
        errors.append("purpose differs from the frozen v1 definition")
    if protocol.get("final_evaluation_data_prohibited") is not True:
        errors.append("final evaluation data must be prohibited")

    prohibited = protocol.get("prohibited_dataset_names")
    if not isinstance(prohibited, list):
        errors.append("prohibited_dataset_names must be a list")
    else:
        normalized = [str(value).casefold() for value in prohibited]
        if normalized != ["scifact", "arguana"]:
            errors.append("frozen final dataset prohibition must remain exact")
        if not FROZEN_FINAL_DATASETS.issubset(normalized):
            errors.append("frozen final datasets must remain explicitly prohibited")

    _require_exact_object(protocol, "input_source", EXPECTED_INPUT_SOURCE, errors)
    _require_exact_object(protocol, "reranker", EXPECTED_RERANKER, errors)
    _require_exact_object(protocol, "matrix", EXPECTED_MATRIX, errors)
    _require_exact_object(protocol, "host_controls", EXPECTED_HOST_CONTROLS, errors)
    _require_exact_object(protocol, "candidate_gates", EXPECTED_CANDIDATE_GATES, errors)
    _require_exact_object(protocol, "claim_boundary", EXPECTED_CLAIM_BOUNDARY, errors)
    return errors


def main() -> int:
    errors = validate(json.loads(PROTOCOL.read_text(encoding="utf-8")))
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(
        "PASS: reranker development protocol exactly matches the frozen v1 "
        "synthetic-only definition"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
