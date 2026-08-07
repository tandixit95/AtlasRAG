#!/usr/bin/env python3
"""Fail-closed validation for the frozen, unexecuted disk benchmark protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA_VERSION = "atlasrag-disk-benchmark-protocol/v1"
PROTOCOL_STATUS = "protocol_unexecuted"
EXPECTED_SCALE_LADDER = [100_000, 1_000_000, 5_000_000, 10_000_000]
DENSE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sum_close_to_one(values: dict[str, Any], name: str) -> None:
    _require(values, f"{name} must not be empty")
    total = sum(float(value) for value in values.values())
    _require(abs(total - 1.0) <= 1e-12, f"{name} must sum to 1.0")
    _require(
        all(float(value) >= 0.0 for value in values.values()),
        f"{name} cannot be negative",
    )


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Validate the protocol's claim boundary and required measurement contract."""

    _require(
        protocol.get("schema_version") == PROTOCOL_SCHEMA_VERSION,
        "unsupported protocol schema",
    )
    _require(
        protocol.get("execution_status") == PROTOCOL_STATUS,
        "disk protocol must remain protocol_unexecuted",
    )
    _require(
        "not evidence" in protocol.get("claim_boundary", "").lower(),
        "claim boundary must state that the protocol is not evidence",
    )

    corpus = protocol["corpus"]
    _require(
        corpus["scale_ladder_documents"] == EXPECTED_SCALE_LADDER,
        "scale ladder must remain 100K, 1M, 5M, 10M",
    )
    _require(
        100_000_000 not in corpus["scale_ladder_documents"],
        "100M cannot enter the v1 executable scale ladder",
    )
    _require(
        corpus["future_100m_status"] == "evaluation_only_unexecuted",
        "100M must remain evaluation-only and unexecuted",
    )
    _require(
        corpus["identity"]["duplicate_identity_policy"] == "fail_closed",
        "duplicate identities must fail closed",
    )
    chunking = corpus["chunking"]
    _require(chunking["chunk_size_chars"] > 0, "chunk size must be positive")
    _require(
        0 <= chunking["overlap_chars"] < chunking["chunk_size_chars"],
        "chunk overlap must be smaller than chunk size",
    )
    _require(
        "chunk_id" in chunking["required_chunk_fields"],
        "chunk identity must be recorded",
    )
    _require(
        "document_content_sha256" in chunking["required_chunk_fields"],
        "document version provenance must be recorded",
    )

    authorization = protocol["authorization"]
    _require(
        authorization["filter_stage"] == "before_scoring_and_ranking",
        "authorization must be applied before scoring and ranking",
    )
    _require(
        authorization["malformed_policy_behavior"] == "fail_closed",
        "malformed authorization policy must fail closed",
    )
    for field in (
        "unauthorized_result_tolerance",
        "excluded_chunk_tolerance",
        "unauthorized_influence_tolerance",
    ):
        _require(authorization[field] == 0, f"{field} must remain zero")

    retrieval = protocol["retrieval"]
    _require(
        retrieval["dense"]["encoder_revision"] == DENSE_REVISION,
        "dense encoder revision must stay pinned",
    )
    _require(
        len(retrieval["dense"]["encoder_revision"]) == 40,
        "dense encoder revision must be a commit SHA",
    )
    _require(
        retrieval["fusion"]["method"] == "reciprocal_rank_fusion",
        "fusion must remain RRF for protocol v1",
    )
    _require(retrieval["top_k"] == 10, "quality cutoff must remain top-10")

    query = protocol["query_workload"]
    _sum_close_to_one(query["class_distribution"], "query class distribution")
    _require(
        query["concurrency_levels"] == [1, 4, 16],
        "concurrency levels must remain [1, 4, 16]",
    )
    _require(
        query["warmup_query_count_per_concurrency"] > 0, "warmup count must be positive"
    )
    _require(
        query["measurement_query_count_per_concurrency"]
        > query["warmup_query_count_per_concurrency"],
        "measurement count must exceed warmup count",
    )
    _require(query["timeout_seconds"] > 0, "query timeout must be positive")

    updates = protocol["update_workload"]
    _sum_close_to_one(updates["event_distribution"], "update event distribution")
    _require(
        updates["resume_checkpoint_interval_events"] > 0,
        "checkpoint interval must be positive",
    )
    for required in (
        "incremental_add",
        "content_update",
        "delete",
        "restart_resume",
        "idempotent_replay",
    ):
        _require(
            required in updates["required_behaviors"],
            f"missing required update behavior: {required}",
        )

    caches = protocol["cache_conditions"]
    _require(
        caches["cold_cache"]["os_page_cache"]
        == "reset_method_must_be_recorded_or_condition_is_inconclusive",
        "cold-cache evidence must fail closed when OS cache reset is unknown",
    )
    _require(
        caches["cold_cache"]["warmup_queries"] == 0,
        "cold-cache condition cannot include warmup queries",
    )
    _require(
        caches["warm_cache"]["warmup_queries"]
        == query["warmup_query_count_per_concurrency"],
        "warm-cache warmup must match query protocol",
    )

    environment = protocol["environment_disclosure"]
    required_environment = {
        "atlasrag_commit",
        "package_version",
        "wheel_sha256",
        "python_version",
        "operating_system",
        "kernel",
        "cpu_model",
        "physical_cores",
        "logical_cores",
        "ram_bytes",
        "storage_model",
        "storage_medium",
        "filesystem",
        "free_disk_bytes_before",
        "free_disk_bytes_after",
        "dense_encoder_name",
        "dense_encoder_revision",
        "dependency_lock_sha256",
    }
    _require(
        required_environment.issubset(set(environment["required_fields"])),
        "environment disclosure is incomplete",
    )
    _require(
        {"credentials", "tokens", "private_paths"}.issubset(
            set(environment["forbidden_fields"])
        ),
        "public artifact privacy exclusions are incomplete",
    )

    measurements = protocol["measurements"]
    for percentile in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
        _require(
            percentile in measurements["query"],
            f"missing latency percentile: {percentile}",
        )
    for required in ("physical_index_bytes", "index_build_seconds", "resume_seconds"):
        _require(
            required in measurements["build"], f"missing build measurement: {required}"
        )
    for required in (
        "authorization_leakage_count",
        "unauthorized_influence_failures",
        "citation_completeness",
    ):
        _require(
            required in measurements["quality"],
            f"missing quality/safety measurement: {required}",
        )

    gates = protocol["quality_gates"]
    _require(
        gates["authorization_leakage_count_max"] == 0,
        "authorization leakage gate must be zero",
    )
    _require(
        gates["excluded_chunk_leakage_count_max"] == 0,
        "excluded chunk leakage gate must be zero",
    )
    _require(
        gates["unauthorized_influence_failures_max"] == 0,
        "unauthorized influence gate must be zero",
    )
    _require(
        gates["citation_completeness_min"] == 1.0, "citation completeness must be 100%"
    )
    _require(
        gates["reproducible_ranking_hashes_required"] is True,
        "ranking reproducibility is required",
    )
    _require(
        gates["resume_equivalence_required"] is True, "resume equivalence is required"
    )
    _require(
        gates["query_error_count_max"] == 0 and gates["query_timeout_count_max"] == 0,
        "errors and timeouts must have zero tolerance",
    )
    _require(
        "report_only" in gates["performance_threshold_policy"],
        "v1 performance metrics must remain report-only, not production SLOs",
    )

    artifacts = protocol["artifact_contract"]
    required_outputs = {
        "environment.json",
        "build.json",
        "query_summary.json",
        "query_samples.csv",
        "quality.json",
        "authorization.json",
        "updates.json",
        "manifest.json",
        "SHA256SUMS",
    }
    _require(
        required_outputs.issubset(set(artifacts["required_outputs"])),
        "raw artifact contract is incomplete",
    )
    _require(artifacts["checksums"] == "sha256", "artifact checksums must use SHA-256")
    _require(
        artifacts["reproduction"]["same_config_runs"] == 2,
        "two equivalent runs are required",
    )
    _require(
        artifacts["reproduction"]["ranking_hashes_must_match"] is True,
        "ranking hashes must reproduce exactly",
    )

    decision = protocol["decision_semantics"]
    for state in ("pass", "fail", "inconclusive", "claim_rule"):
        _require(bool(decision.get(state)), f"decision semantics missing: {state}")
    _require(
        "measured" in decision["claim_rule"].lower(),
        "claim rule must require measured evidence",
    )

    scale_policy = protocol["scale_execution_policy"]
    required_large_scale_policy = (
        "requires_pre_run_runtime_disk_ram_gpu_cost_estimate_and_explicit_approval"
    )
    for scale in ("1000000", "5000000", "10000000"):
        _require(
            scale_policy[scale] == required_large_scale_policy,
            (
                f"{int(scale) // 1_000_000}M execution must require an estimate "
                "and explicit approval"
            ),
        )
    _require(
        "not_in_v1_scale_ladder" in scale_policy["100000000"],
        "100M must remain outside protocol v1 execution",
    )


def load_and_validate(path: Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def main() -> int:
    path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "disk-backed-v1-unexecuted.json"
    )
    protocol = load_and_validate(path)
    print(
        json.dumps(
            {
                "status": "PASS",
                "protocol_id": protocol["protocol_id"],
                "execution_status": protocol["execution_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
