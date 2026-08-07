from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.scale.src.verify_disk_benchmark_protocol import validate_protocol

PROTOCOL = (
    Path(__file__).resolve().parents[1] / "protocols" / "disk-backed-v1-unexecuted.json"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_frozen_disk_benchmark_protocol_is_valid() -> None:
    validate_protocol(_protocol())


def test_protocol_cannot_be_relabelled_as_executed_evidence() -> None:
    protocol = _protocol()
    protocol["execution_status"] = "measured_scale"

    with pytest.raises(ValueError, match="protocol_unexecuted"):
        validate_protocol(protocol)


def test_protocol_keeps_100m_outside_v1_execution_ladder() -> None:
    protocol = _protocol()
    protocol["corpus"]["scale_ladder_documents"].append(100_000_000)

    with pytest.raises(ValueError, match="scale ladder"):
        validate_protocol(protocol)


def test_authorization_cannot_move_after_ranking() -> None:
    protocol = _protocol()
    protocol["authorization"]["filter_stage"] = "after_ranking"

    with pytest.raises(ValueError, match="before scoring and ranking"):
        validate_protocol(protocol)


def test_protocol_requires_p99_and_raw_evidence_contract() -> None:
    protocol = _protocol()
    protocol["measurements"]["query"].remove("latency_p99_ms")

    with pytest.raises(ValueError, match="latency_p99_ms"):
        validate_protocol(protocol)


def test_protocol_requires_pinned_dense_model_revision() -> None:
    protocol = _protocol()
    protocol["retrieval"]["dense"]["encoder_revision"] = "main"

    with pytest.raises(ValueError, match="dense encoder revision"):
        validate_protocol(protocol)


def test_protocol_rejects_nonzero_authorization_tolerance() -> None:
    protocol = _protocol()
    protocol["quality_gates"]["authorization_leakage_count_max"] = 1

    with pytest.raises(ValueError, match="authorization leakage gate"):
        validate_protocol(protocol)


def test_protocol_query_distribution_must_sum_to_one() -> None:
    protocol = _protocol()
    protocol["query_workload"]["class_distribution"]["positive_exact_lookup"] = 0.79

    with pytest.raises(ValueError, match="query class distribution"):
        validate_protocol(protocol)


def test_protocol_mutation_fixture_does_not_modify_source() -> None:
    original = _protocol()
    mutated = copy.deepcopy(original)
    mutated["retrieval"]["top_k"] = 20

    with pytest.raises(ValueError, match="top-10"):
        validate_protocol(mutated)
    validate_protocol(original)
