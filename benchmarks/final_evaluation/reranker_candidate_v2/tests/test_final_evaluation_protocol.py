from __future__ import annotations

import copy
import hashlib
import json

from benchmarks.final_evaluation.reranker_candidate_v2 import verify_protocol as module


def fixtures():
    profile_bytes = module.PROFILE.read_bytes()
    return (
        json.loads(module.PROTOCOL.read_text()),
        json.loads(module.GATES.read_text()),
        json.loads(profile_bytes),
        hashlib.sha256(profile_bytes).hexdigest(),
    )


def validate(protocol, gates, profile, profile_sha):
    return module.validate(protocol, gates, profile, profile_sha256=profile_sha)


def test_current_freeze_passes():
    protocol, gates, profile, profile_sha = fixtures()
    assert validate(protocol, gates, profile, profile_sha) == []


def test_profile_hash_mismatch_fails_closed():
    protocol, gates, profile, _ = fixtures()
    assert any(
        "SHA-256" in error for error in validate(protocol, gates, profile, "0" * 64)
    )


def test_profile_candidate_change_fails_closed():
    protocol, gates, profile, profile_sha = fixtures()
    profile = copy.deepcopy(profile)
    profile["nominated_candidates"][0]["batch_size"] = 64
    assert any(
        "nomination" in error
        for error in validate(protocol, gates, profile, profile_sha)
    )


def test_protocol_batch_change_fails_closed():
    protocol, gates, profile, profile_sha = fixtures()
    protocol = copy.deepcopy(protocol)
    protocol["candidate"]["reranker_batch_size"] = 64
    assert any(
        "batch size" in error
        for error in validate(protocol, gates, profile, profile_sha)
    )


def test_execution_batch_change_fails_closed():
    protocol, gates, profile, profile_sha = fixtures()
    protocol = copy.deepcopy(protocol)
    protocol["execution"]["reranker_batch_size"] = 64
    assert any(
        "execution" in error
        for error in validate(protocol, gates, profile, profile_sha)
    )


def test_task_shape_change_fails_closed():
    protocol, gates, profile, profile_sha = fixtures()
    protocol = copy.deepcopy(protocol)
    protocol["required_tasks"][0]["query_count"] = 1
    assert any(
        "task shapes" in error
        for error in validate(protocol, gates, profile, profile_sha)
    )


def test_batch_veto_removal_fails_closed():
    protocol, gates, profile, profile_sha = fixtures()
    gates = copy.deepcopy(gates)
    gates["per_task_vetoes"] = [
        item
        for item in gates["per_task_vetoes"]
        if item["id"] != "reranker_batch_size_frozen"
    ]
    assert any(
        "batch-size veto" in error
        for error in validate(protocol, gates, profile, profile_sha)
    )


def test_p95_budget_change_fails_closed():
    protocol, gates, profile, profile_sha = fixtures()
    gates = copy.deepcopy(gates)
    next(
        item
        for item in gates["per_task_vetoes"]
        if item["id"] == "reranker_p95_budget_ms"
    )["threshold"] = 1000.0
    assert any(
        "75 ms" in error for error in validate(protocol, gates, profile, profile_sha)
    )


def test_post_outcome_tuning_enable_fails_closed():
    protocol, gates, profile, profile_sha = fixtures()
    protocol = copy.deepcopy(protocol)
    protocol["outcome_handling"]["post_outcome_candidate_tuning_allowed"] = True
    assert any(
        "outcome handling" in error
        for error in validate(protocol, gates, profile, profile_sha)
    )


def test_default_promotion_before_execution_fails_closed():
    protocol, gates, profile, profile_sha = fixtures()
    protocol = copy.deepcopy(protocol)
    protocol["claim_boundary"]["default_promotion_allowed_before_execution"] = True
    assert any(
        "claim boundary" in error
        for error in validate(protocol, gates, profile, profile_sha)
    )
