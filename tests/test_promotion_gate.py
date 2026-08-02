from __future__ import annotations

from copy import deepcopy

from atlasrag.evaluation import (
    PromotionDisposition,
    evaluate_promotion,
)


def _policy() -> dict:
    return {
        "required_task_shapes": ["one", "two"],
        "global_vetoes": [
            {
                "id": "leakage",
                "path": ["contracts", "leakage"],
                "operator": "eq",
                "threshold": 0,
            }
        ],
        "per_task_vetoes": [
            {
                "id": "recall",
                "path": ["quality", "recall", "mean_delta"],
                "operator": "ge",
                "threshold": 0.0,
            }
        ],
        "per_task_promotion_gates": [
            {
                "id": "mrr_interval",
                "path": ["quality", "mrr", "interval", 0],
                "operator": "gt",
                "threshold": 0.0,
            }
        ],
    }


def _evidence() -> dict:
    task = {
        "quality": {
            "recall": {"mean_delta": 0.0},
            "mrr": {"interval": [0.001, 0.02]},
        }
    }
    return {
        "contracts": {"leakage": 0},
        "tasks": {"one": deepcopy(task), "two": deepcopy(task)},
    }


def test_all_gates_pass_promotes_candidate() -> None:
    decision = evaluate_promotion(_policy(), _evidence())

    assert decision.disposition is PromotionDisposition.PROMOTE
    assert decision.enable_candidate_by_default
    assert decision.passed


def test_observed_veto_violation_rejects_candidate() -> None:
    evidence = _evidence()
    evidence["tasks"]["two"]["quality"]["recall"]["mean_delta"] = -0.01

    decision = evaluate_promotion(_policy(), evidence)

    assert decision.disposition is PromotionDisposition.RETAIN_DEFAULT_REJECTED
    assert not decision.enable_candidate_by_default


def test_unproven_primary_improvement_is_inconclusive() -> None:
    evidence = _evidence()
    evidence["tasks"]["one"]["quality"]["mrr"]["interval"][0] = -0.001

    decision = evaluate_promotion(_policy(), evidence)

    assert decision.disposition is PromotionDisposition.RETAIN_DEFAULT_INCONCLUSIVE
    assert not decision.enable_candidate_by_default


def test_missing_task_evidence_fails_closed_as_inconclusive() -> None:
    evidence = _evidence()
    del evidence["tasks"]["two"]

    decision = evaluate_promotion(_policy(), evidence)

    assert decision.disposition is PromotionDisposition.RETAIN_DEFAULT_INCONCLUSIVE
    assert decision.missing_count == 2
    assert not decision.enable_candidate_by_default
