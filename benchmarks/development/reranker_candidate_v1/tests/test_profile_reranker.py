from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.development.reranker_candidate_v1 import profile_reranker as module


class DeterministicPredictor:
    def predict(self, pairs, *, batch_size):
        del batch_size
        return [
            float(sum(ord(char) for char in query + document) % 997)
            for query, document in pairs
        ]


class WrongCountPredictor:
    def predict(self, pairs, *, batch_size):
        del batch_size
        return [1.0] * max(0, len(pairs) - 1)


class NonFinitePredictor:
    def predict(self, pairs, *, batch_size):
        del batch_size
        return [math.inf] * len(pairs)


class FlippingPredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, pairs, *, batch_size):
        del batch_size
        self.calls += 1
        values = list(range(len(pairs)))
        if self.calls % 2:
            values.reverse()
        return [float(value) for value in values]


def protocol():
    return module.load_protocol()


def test_generated_workload_is_deterministic_and_final_dataset_free():
    first = module.generate_synthetic_inputs(protocol())
    second = module.generate_synthetic_inputs(protocol())
    assert first == second
    assert len(first) == 32
    assert len(first[0]["documents"]) == 50
    payload = json.dumps(first).casefold()
    assert "scifact" not in payload
    assert "arguana" not in payload
    assert module.workload_sha256(first) == module.workload_sha256(second)


def test_score_count_and_finiteness_fail_closed():
    rows = module.generate_synthetic_inputs(protocol())[:1]
    pairs = module.build_pairs(rows, 10)
    with pytest.raises(RuntimeError, match="score count mismatch"):
        scores = module._coerce_scores(
            WrongCountPredictor().predict(pairs, batch_size=16)
        )
        module.validate_scores(scores, len(pairs))
    with pytest.raises(RuntimeError, match="non-finite"):
        scores = module._coerce_scores(
            NonFinitePredictor().predict(pairs, batch_size=16)
        )
        module.validate_scores(scores, len(pairs))


def test_ordering_digest_is_repeatable():
    rows = module.generate_synthetic_inputs(protocol())[:2]
    pairs = module.build_pairs(rows, 10)
    predictor = DeterministicPredictor()
    first = predictor.predict(pairs, batch_size=16)
    second = predictor.predict(pairs, batch_size=64)
    assert module.ordering_digest(rows, first, 10) == module.ordering_digest(
        rows, second, 10
    )


def test_profile_configuration_records_raw_samples_and_determinism():
    rows = module.generate_synthetic_inputs(protocol())[:2]
    result = module.profile_configuration(
        DeterministicPredictor(),
        rows,
        depth=10,
        batch_size=16,
        warmups=1,
        measured_iterations=3,
        synchronize=lambda: None,
    )
    assert result["repeat_order_deterministic"] is True
    assert result["pair_count_per_iteration"] == 20
    assert len(result["raw_timing_ms"]) == 3
    assert result["timing_ms"]["p95"] >= 0.0


def test_profile_configuration_rejects_order_instability():
    rows = module.generate_synthetic_inputs(protocol())[:1]
    with pytest.raises(RuntimeError, match="ordering determinism"):
        module.profile_configuration(
            FlippingPredictor(),
            rows,
            depth=10,
            batch_size=16,
            warmups=0,
            measured_iterations=1,
            synchronize=lambda: None,
        )


def test_candidate_nomination_stays_at_frozen_default_depth_and_is_unique():
    value = protocol()
    results = [
        {
            "candidate_depth": 10,
            "batch_size": 16,
            "repeat_order_deterministic": True,
            "peak_accelerator_memory_mib": 100.0,
            "timing_ms": {"p95": 105.0},
        },
        {
            "candidate_depth": 10,
            "batch_size": 32,
            "repeat_order_deterministic": True,
            "peak_accelerator_memory_mib": 100.0,
            "timing_ms": {"p95": 100.0},
        },
        {
            "candidate_depth": 50,
            "batch_size": 64,
            "repeat_order_deterministic": True,
            "peak_accelerator_memory_mib": 100.0,
            "timing_ms": {"p95": 1.0},
        },
    ]
    candidate = module.nominate_candidate(results, value)
    assert candidate is not None
    assert candidate["candidate_depth"] == 10
    assert candidate["batch_size"] == 16


def test_memory_gate_can_leave_no_candidate():
    value = protocol()
    results = [
        {
            "candidate_depth": 10,
            "batch_size": 16,
            "repeat_order_deterministic": True,
            "peak_accelerator_memory_mib": 7000.0,
            "timing_ms": {"p95": 1.0},
        }
    ]
    assert module.nominate_candidate(results, value) is None


def test_exclusive_lock_rejects_second_profiler(tmp_path: Path):
    lock = tmp_path / "profiler.lock"
    with module.exclusive_benchmark_lock(lock):
        with pytest.raises(RuntimeError, match="already held"):
            with module.exclusive_benchmark_lock(lock):
                pass
    assert not lock.exists()


def test_offline_environment_is_applied_and_restored(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "original")
    with module.offline_model_environment():
        assert module.os.environ["HF_HUB_OFFLINE"] == "1"
        assert module.os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert module.os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in module.os.environ
    assert module.os.environ["TRANSFORMERS_OFFLINE"] == "original"


def test_expected_snapshot_path_pins_revision(monkeypatch, tmp_path: Path):
    value = protocol()
    monkeypatch.setattr(module, "_hub_root", lambda: tmp_path)
    snapshot = module.expected_snapshot_path(value)
    assert snapshot.name == value["reranker"]["revision"]
    assert snapshot.parent.parent.name == (
        "models--cross-encoder--ms-marco-MiniLM-L6-v2"
    )


def test_missing_snapshot_fails_closed(monkeypatch, tmp_path: Path):
    value = protocol()
    monkeypatch.setattr(module, "_hub_root", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="pinned local model snapshot is missing"):
        module.require_local_snapshot(value)


def test_real_loader_fails_closed_without_optional_dependency(
    monkeypatch, tmp_path: Path
):
    if module.package_version("sentence-transformers") is not None:
        pytest.skip("optional reranking dependency is installed")
    value = protocol()
    snapshot = tmp_path / value["reranker"]["revision"]
    snapshot.mkdir()
    monkeypatch.setattr(module, "expected_snapshot_path", lambda _protocol: snapshot)
    with pytest.raises(RuntimeError, match="reranking.*extra"):
        module.load_cross_encoder(value, "cpu")
