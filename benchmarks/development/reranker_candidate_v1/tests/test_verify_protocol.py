from __future__ import annotations

import copy
import json

from benchmarks.development.reranker_candidate_v1 import verify_protocol as module


def valid_protocol():
    return json.loads(module.PROTOCOL.read_text())


def test_current_protocol_passes():
    assert module.validate(valid_protocol()) == []


def test_final_dataset_payload_use_fails_closed():
    protocol = copy.deepcopy(valid_protocol())
    protocol["input_source"]["payload_from_final_evaluation_sets"] = True
    assert any("input_source" in error for error in module.validate(protocol))


def test_missing_frozen_dataset_prohibition_fails_closed():
    protocol = copy.deepcopy(valid_protocol())
    protocol["prohibited_dataset_names"] = ["scifact"]
    assert any("prohibition" in error for error in module.validate(protocol))


def test_default_promotion_claim_fails_closed():
    protocol = copy.deepcopy(valid_protocol())
    protocol["claim_boundary"]["default_promotion_allowed"] = True
    assert any("claim_boundary" in error for error in module.validate(protocol))


def test_synthetic_seed_is_frozen():
    protocol = copy.deepcopy(valid_protocol())
    protocol["input_source"]["seed"] += 1
    assert any("input_source" in error for error in module.validate(protocol))


def test_template_version_is_frozen():
    protocol = copy.deepcopy(valid_protocol())
    protocol["input_source"]["document_template_version"] = "other"
    assert any("input_source" in error for error in module.validate(protocol))


def test_batch_matrix_is_frozen():
    protocol = copy.deepcopy(valid_protocol())
    protocol["matrix"]["batch_sizes"] = [1]
    assert any("matrix" in error for error in module.validate(protocol))


def test_default_depth_is_frozen():
    protocol = copy.deepcopy(valid_protocol())
    protocol["candidate_gates"]["default_candidate_depth"] = 50
    assert any("candidate_gates" in error for error in module.validate(protocol))


def test_memory_cap_is_required_and_frozen():
    protocol = copy.deepcopy(valid_protocol())
    del protocol["candidate_gates"]["peak_accelerator_memory_mib_max"]
    assert any("candidate_gates" in error for error in module.validate(protocol))


def test_p95_ratio_must_match_frozen_positive_value():
    protocol = copy.deepcopy(valid_protocol())
    protocol["candidate_gates"]["batch_size_p95_over_best_same_depth_max_ratio"] = -1
    assert any("candidate_gates" in error for error in module.validate(protocol))
