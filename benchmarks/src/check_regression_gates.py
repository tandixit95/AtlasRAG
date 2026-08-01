from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check(label: str, actual: Any, comparator: str, threshold: Any) -> dict[str, Any]:
    if comparator == ">=":
        passed = actual >= threshold
    elif comparator == "<=":
        passed = actual <= threshold
    elif comparator == "==":
        passed = actual == threshold
    else:
        raise ValueError(comparator)
    return {
        "gate": label,
        "actual": actual,
        "comparator": comparator,
        "threshold": threshold,
        "passed": passed,
    }


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gates", type=Path, required=True)
    p.add_argument("--scifact", type=Path, required=True)
    p.add_argument("--arguana", type=Path, required=True)
    p.add_argument("--safety", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--reproducibility", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    gates = load(args.gates)
    public = {
        "scifact": load(args.scifact),
        "arguana-contrast-200": load(args.arguana),
    }
    safety = load(args.safety)
    adapter = load(args.adapter)
    repro = load(args.reproducibility)
    checks: list[dict[str, Any]] = []

    for dataset, requirements in gates["public_quality_minimums"].items():
        report = public[dataset]
        for method, metric_requirements in requirements.items():
            if method == "ann_mean_top10_overlap":
                checks.append(
                    check(
                        f"{dataset}.ann_mean_top10_overlap",
                        report["ann_vs_exact"]["mean_top10_overlap"],
                        ">=",
                        metric_requirements,
                    )
                )
                continue
            for metric, threshold in metric_requirements.items():
                checks.append(
                    check(
                        f"{dataset}.{method}.{metric}",
                        report["quality"][method][metric],
                        ">=",
                        threshold,
                    )
                )

        systems = gates["local_system_maximums"]
        checks.append(
            check(
                f"{dataset}.hybrid_sequential_p95_ms",
                report["sampled_latency_ms"]["hybrid_sequential"]["p95"],
                "<=",
                systems["hybrid_sequential_p95_ms"],
            )
        )
        checks.append(
            check(
                f"{dataset}.process_peak_rss_kb",
                report["systems"]["process_peak_rss_kb"],
                "<=",
                systems["process_peak_rss_kb"],
            )
        )
        checks.append(
            check(
                f"{dataset}.serialized_index_artifact_bytes",
                report["build"]["artifact_bytes"]["total"],
                "<=",
                systems["serialized_index_artifact_bytes"],
            )
        )

    for metric, threshold in gates["safety_requirements"].items():
        checks.append(
            check(f"safety.{metric}", safety["metrics"][metric], "==", threshold)
        )
    for metric, threshold in gates["adapter_requirements"].items():
        checks.append(
            check(f"adapter.{metric}", adapter["metrics"][metric], "==", threshold)
        )
    checks.append(
        check(
            "reproducibility.passed",
            repro["passed"],
            "==",
            gates["reproducibility_required"],
        )
    )

    report = {
        "schema_version": "atlasrag.regression-gate-report.v1",
        "scope": gates["scope"],
        "passed": all(row["passed"] for row in checks),
        "check_count": len(checks),
        "failed_count": sum(not row["passed"] for row in checks),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
