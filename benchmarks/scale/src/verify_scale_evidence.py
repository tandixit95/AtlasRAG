#!/usr/bin/env python3
"""Validate a committed AtlasRAG scale-evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.scale.src.scale_harness import (
    ScaleConfig,
    compare_reproducibility,
    evidence_from_mapping,
    validate_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    if bundle.get("bundle_schema_version") != "atlasrag-scale-evidence-bundle/v1":
        raise SystemExit("unsupported bundle schema")
    config = ScaleConfig.from_mapping(bundle["config"])
    if config.execution_status == "target_unexecuted":
        raise SystemExit("target configurations are not measured evidence")
    first = evidence_from_mapping(bundle["first_run"])
    second = evidence_from_mapping(bundle["second_run"])
    validate_evidence(config, first)
    validate_evidence(config, second)
    comparison = compare_reproducibility(first, second)
    if not comparison["exact_reproduction"]:
        raise SystemExit("bundle does not reproduce exactly")
    if comparison != bundle["reproducibility"]:
        raise SystemExit("stored reproducibility report does not match evidence")
    print(json.dumps({"status": "PASS", "bundle": str(args.bundle)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
