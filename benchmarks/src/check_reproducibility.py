from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(dataset: str, a_path: Path, b_path: Path) -> dict[str, Any]:
    a = load(a_path)
    b = load(b_path)
    a_raw = a_path.parent / a["artifacts"]["raw_queries"]
    b_raw = b_path.parent / b["artifacts"]["raw_queries"]
    a_hash = sha256(a_raw)
    b_hash = sha256(b_raw)
    return {
        "dataset": dataset,
        "quality_equal": a["quality"] == b["quality"],
        "run_parameters_equal": a["run"] == b["run"],
        "dataset_hashes_equal": a["dataset"] == b["dataset"],
        "model_revision_equal": a["model"]["cached_revision"]
        == b["model"]["cached_revision"],
        "raw_rankings_equal": a_hash == b_hash,
        "raw_rankings_sha256": a_hash,
        "quality": a["quality"],
        "note": "Quality, rankings, run parameters, frozen inputs, model revision, and deterministic sample IDs must match exactly. Latency and build timing may vary.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scifact-a", type=Path, required=True)
    p.add_argument("--scifact-b", type=Path, required=True)
    p.add_argument("--arguana-a", type=Path, required=True)
    p.add_argument("--arguana-b", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows = [
        compare("scifact", args.scifact_a, args.scifact_b),
        compare("arguana-contrast-200", args.arguana_a, args.arguana_b),
    ]
    required = (
        "quality_equal",
        "run_parameters_equal",
        "dataset_hashes_equal",
        "model_revision_equal",
        "raw_rankings_equal",
    )
    report = {
        "schema_version": "atlasrag.reproducibility.v1",
        "passed": all(all(row[key] for key in required) for row in rows),
        "datasets": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
