"""Fail-closed evaluation of machine-readable default-path promotion gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class GateClass(StrEnum):
    VETO = "veto"
    PROMOTION = "promotion"


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    MISSING = "missing"


class PromotionDisposition(StrEnum):
    PROMOTE = "promote"
    RETAIN_DEFAULT_REJECTED = "retain_default_rejected"
    RETAIN_DEFAULT_INCONCLUSIVE = "retain_default_inconclusive"


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    gate_class: GateClass
    status: GateStatus
    operator: str
    threshold: Any
    actual: Any = None
    task_id: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    disposition: PromotionDisposition
    enable_candidate_by_default: bool
    passed: bool
    check_count: int
    failed_count: int
    missing_count: int
    checks: tuple[GateResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "enable_candidate_by_default": self.enable_candidate_by_default,
            "passed": self.passed,
            "check_count": self.check_count,
            "failed_count": self.failed_count,
            "missing_count": self.missing_count,
            "checks": [
                {
                    **asdict(check),
                    "gate_class": check.gate_class.value,
                    "status": check.status.value,
                }
                for check in self.checks
            ],
        }


def _read_path(value: Any, path: Sequence[Any]) -> tuple[bool, Any]:
    current = value
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, Sequence) or isinstance(
                current, (str, bytes, bytearray)
            ):
                return False, None
            if part < 0 or part >= len(current):
                return False, None
            current = current[part]
            continue
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _compare(actual: Any, operator: str, threshold: Any) -> bool:
    if operator == "eq":
        return actual == threshold
    if operator == "ge":
        return actual >= threshold
    if operator == "gt":
        return actual > threshold
    if operator == "le":
        return actual <= threshold
    if operator == "lt":
        return actual < threshold
    raise ValueError(f"unsupported gate operator: {operator}")


def _evaluate_gate(
    gate: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    gate_class: GateClass,
    task_id: str | None = None,
) -> GateResult:
    gate_id = str(gate["id"])
    path = gate["path"]
    if not isinstance(path, list):
        raise TypeError(f"gate {gate_id} path must be a list")
    operator = str(gate["operator"])
    threshold = gate["threshold"]
    found, actual = _read_path(evidence, path)
    if not found:
        return GateResult(
            gate_id=gate_id,
            gate_class=gate_class,
            status=GateStatus.MISSING,
            operator=operator,
            threshold=threshold,
            task_id=task_id,
            detail="required evidence path is missing",
        )
    try:
        passed = _compare(actual, operator, threshold)
    except (TypeError, ValueError) as exc:
        return GateResult(
            gate_id=gate_id,
            gate_class=gate_class,
            status=GateStatus.FAIL,
            operator=operator,
            threshold=threshold,
            actual=actual,
            task_id=task_id,
            detail=f"comparison failed: {exc}",
        )
    return GateResult(
        gate_id=gate_id,
        gate_class=gate_class,
        status=GateStatus.PASS if passed else GateStatus.FAIL,
        operator=operator,
        threshold=threshold,
        actual=actual,
        task_id=task_id,
    )


def evaluate_promotion(
    policy: Mapping[str, Any], evidence: Mapping[str, Any]
) -> PromotionDecision:
    """Evaluate every frozen gate and retain the default on any uncertainty."""

    required_tasks = policy.get("required_task_shapes")
    if not isinstance(required_tasks, list) or not required_tasks:
        raise ValueError("policy requires at least one task shape")
    tasks = evidence.get("tasks")
    if not isinstance(tasks, Mapping):
        tasks = {}

    checks: list[GateResult] = []
    for gate in policy.get("global_vetoes", []):
        checks.append(
            _evaluate_gate(
                gate,
                evidence,
                gate_class=GateClass.VETO,
            )
        )

    for task_id in required_tasks:
        task_evidence = tasks.get(task_id)
        if not isinstance(task_evidence, Mapping):
            for gate_class, gate_key in (
                (GateClass.VETO, "per_task_vetoes"),
                (GateClass.PROMOTION, "per_task_promotion_gates"),
            ):
                for gate in policy.get(gate_key, []):
                    checks.append(
                        GateResult(
                            gate_id=str(gate["id"]),
                            gate_class=gate_class,
                            status=GateStatus.MISSING,
                            operator=str(gate["operator"]),
                            threshold=gate["threshold"],
                            task_id=str(task_id),
                            detail="required task evidence is missing",
                        )
                    )
            continue

        for gate in policy.get("per_task_vetoes", []):
            checks.append(
                _evaluate_gate(
                    gate,
                    task_evidence,
                    gate_class=GateClass.VETO,
                    task_id=str(task_id),
                )
            )
        for gate in policy.get("per_task_promotion_gates", []):
            checks.append(
                _evaluate_gate(
                    gate,
                    task_evidence,
                    gate_class=GateClass.PROMOTION,
                    task_id=str(task_id),
                )
            )

    observed_veto_failure = any(
        check.gate_class is GateClass.VETO and check.status is GateStatus.FAIL
        for check in checks
    )
    any_missing = any(check.status is GateStatus.MISSING for check in checks)
    any_promotion_failure = any(
        check.gate_class is GateClass.PROMOTION and check.status is GateStatus.FAIL
        for check in checks
    )
    all_pass = bool(checks) and all(check.status is GateStatus.PASS for check in checks)

    if all_pass:
        disposition = PromotionDisposition.PROMOTE
    elif observed_veto_failure:
        disposition = PromotionDisposition.RETAIN_DEFAULT_REJECTED
    elif any_missing or any_promotion_failure:
        disposition = PromotionDisposition.RETAIN_DEFAULT_INCONCLUSIVE
    else:
        disposition = PromotionDisposition.RETAIN_DEFAULT_INCONCLUSIVE

    return PromotionDecision(
        disposition=disposition,
        enable_candidate_by_default=disposition is PromotionDisposition.PROMOTE,
        passed=disposition is PromotionDisposition.PROMOTE,
        check_count=len(checks),
        failed_count=sum(check.status is GateStatus.FAIL for check in checks),
        missing_count=sum(check.status is GateStatus.MISSING for check in checks),
        checks=tuple(checks),
    )
