from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.src.analyze_promotion_rank_movements import (
    analyze_rows,
    contains_forbidden_text_key,
    load_rows,
    movement_class,
    sha256,
)

RANKINGS = Path("benchmarks/promotion/artifacts/arguana-rankings.jsonl.gz")


def _row() -> dict:
    return {
        "query_id": "q1",
        "relevant_ids": ["d1"],
        "baseline_top": [f"d{index}" for index in range(1, 11)],
        "reranked_top": [
            {
                "external_id": f"d{index}",
                "candidate_rank": index,
                "reranker_score": float(11 - index),
            }
            for index in range(1, 11)
        ],
        "baseline_metrics": {
            "recall@10": 1.0,
            "mrr@10": 1.0,
            "ndcg@10": 1.0,
            "success@10": 1.0,
        },
        "candidate_metrics": {
            "recall@10": 1.0,
            "mrr@10": 1.0,
            "ndcg@10": 1.0,
            "success@10": 1.0,
        },
    }


def test_movement_class_covers_all_states() -> None:
    assert movement_class(2, 1) == "improved"
    assert movement_class(2, 2) == "unchanged"
    assert movement_class(2, 3) == "regressed"
    assert movement_class(None, 3) == "rescued"
    assert movement_class(3, None) == "lost"
    assert movement_class(None, None) == "both_absent"


def test_analysis_rejects_candidate_set_changes() -> None:
    row = _row()
    row["reranked_top"][-1]["external_id"] = "other"
    with pytest.raises(ValueError, match="candidate set changed"):
        analyze_rows([row], source_path="fixture", source_sha256="0" * 64)


def test_forbidden_payload_text_keys_fail_closed() -> None:
    row = _row()
    row["query_text"] = "not allowed"
    assert contains_forbidden_text_key(row)


def test_plain_jsonl_input_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "rankings.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    assert load_rows(path) == [_row()]


def test_boolean_numeric_evidence_fails_closed() -> None:
    row = _row()
    row["reranked_top"][0]["reranker_score"] = True
    with pytest.raises(ValueError, match="numeric reranker_score"):
        analyze_rows([row], source_path="fixture", source_sha256="0" * 64)


def test_real_arguana_analysis_matches_frozen_evidence() -> None:
    rows = load_rows(RANKINGS)
    report = analyze_rows(
        rows,
        source_path=RANKINGS.as_posix(),
        source_sha256=sha256(RANKINGS),
    )
    assert report["source"]["query_count"] == 200
    assert report["source"]["candidate_set_preserved_queries"] == 200
    assert report["classification_counts"] == {
        "improved": 45,
        "unchanged": 44,
        "regressed": 80,
        "rescued": 0,
        "lost": 0,
        "both_absent": 31,
    }
    assert report["metric_movements"]["mrr@10"] == {
        "mean_delta": -0.06093452381,
        "sum_delta": -12.186904761905,
        "improved_queries": 45,
        "degraded_queries": 80,
        "unchanged_queries": 75,
    }
    assert report["metric_movements"]["recall@10"]["mean_delta"] == 0.0
    assert report["rank_movements"]["mean_rank_delta"] == 0.579881656805
    assert report["rank_movements"]["minimum_rank_delta"] == -9
    assert report["rank_movements"]["maximum_rank_delta"] == 8
    assert report["rank_movements"]["by_baseline_rank"]["1"] == {
        "query_count": 52,
        "improved": 0,
        "unchanged": 24,
        "regressed": 28,
        "mean_rank_delta": 1.346153846154,
    }
    assert not contains_forbidden_text_key(report)


def test_analysis_is_deterministic() -> None:
    rows = load_rows(RANKINGS)
    first = analyze_rows(
        rows,
        source_path=RANKINGS.as_posix(),
        source_sha256=sha256(RANKINGS),
    )
    second = analyze_rows(
        copy.deepcopy(rows),
        source_path=RANKINGS.as_posix(),
        source_sha256=sha256(RANKINGS),
    )
    assert first == second
