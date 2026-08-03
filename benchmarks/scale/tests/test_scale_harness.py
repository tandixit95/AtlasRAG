from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.scale.src.scale_harness import (
    SCHEMA_VERSION,
    ScaleConfig,
    compare_reproducibility,
    execute_harness,
    generate_document,
    validate_evidence,
)


def _config(**overrides: object) -> ScaleConfig:
    values: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "test-smoke",
        "execution_status": "measured_smoke",
        "seed": "test-seed",
        "document_count": 100,
        "shard_count": 8,
        "tenant_count": 4,
        "groups_per_tenant": 3,
        "public_fraction": 0.25,
        "min_text_bytes": 128,
        "max_text_bytes": 256,
        "query_count": 25,
        "update_fraction": 0.1,
        "materialize_corpus": False,
    }
    values.update(overrides)
    return ScaleConfig.from_mapping(values)


def test_document_generation_is_deterministic_and_bounded() -> None:
    config = _config()
    first = generate_document(config, 17)
    second = generate_document(config, 17)

    assert first == second
    assert first.document_id == "doc-000000000017"
    assert "needle_000000000017" in first.text
    assert config.min_text_bytes <= len(first.text.encode()) <= config.max_text_bytes
    assert 0 <= first.shard_id < config.shard_count


def test_measured_smoke_reproduces_exact_claim_surface() -> None:
    config = _config()
    first = execute_harness(config)
    second = execute_harness(config)
    validate_evidence(config, first)
    validate_evidence(config, second)

    report = compare_reproducibility(first, second)

    assert report["exact_reproduction"] is True
    assert first.document_count == 100
    assert sum(first.shard_document_counts) == 100
    assert first.update_count == 10
    assert first.corpus_stream_sha256 == second.corpus_stream_sha256


def test_seed_change_changes_corpus_and_workload_digests() -> None:
    first = execute_harness(_config(seed="first"))
    second = execute_harness(_config(seed="second"))

    assert first.corpus_stream_sha256 != second.corpus_stream_sha256
    assert first.query_workload_sha256 != second.query_workload_sha256


def test_materialized_file_digest_matches_stream(tmp_path) -> None:
    config = _config(materialize_corpus=True)
    evidence = execute_harness(config, tmp_path)
    validate_evidence(config, evidence)

    assert evidence.materialized_corpus is True
    assert evidence.materialized_file_sha256 == evidence.corpus_stream_sha256
    assert (tmp_path / "synthetic-corpus.jsonl").exists()


def test_unexecuted_target_cannot_produce_or_validate_evidence() -> None:
    target = _config(execution_status="target_unexecuted")
    measured = execute_harness(_config())

    with pytest.raises(ValueError, match="cannot produce results"):
        execute_harness(target)
    with pytest.raises(ValueError, match="cannot have claimable evidence"):
        validate_evidence(target, measured)


def test_validation_rejects_count_and_digest_mismatch() -> None:
    config = _config()
    evidence = execute_harness(config)

    with pytest.raises(ValueError, match="document count"):
        validate_evidence(config, replace(evidence, document_count=99))
    with pytest.raises(ValueError, match="SHA-256"):
        validate_evidence(config, replace(evidence, corpus_stream_sha256="bad"))


def test_invalid_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="shard_count"):
        _config(document_count=2, shard_count=3)
    with pytest.raises(ValueError, match="text byte bounds"):
        _config(min_text_bytes=100, max_text_bytes=99)
    with pytest.raises(ValueError, match="unexecuted target"):
        _config(execution_status="target_unexecuted", materialize_corpus=True)
