"""Fail-closed evaluation and default-path promotion contracts."""

from atlasrag.evaluation.promotion import (
    GateClass,
    GateResult,
    GateStatus,
    PromotionDecision,
    PromotionDisposition,
    evaluate_promotion,
)

__all__ = [
    "GateClass",
    "GateResult",
    "GateStatus",
    "PromotionDecision",
    "PromotionDisposition",
    "evaluate_promotion",
]
