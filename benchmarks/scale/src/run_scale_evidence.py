#!/usr/bin/env python3
"""Run AtlasRAG synthetic scale evidence twice and write a validated bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.scale.src.scale_harness import (
    ScaleConfig,
    compare_reproducibility,
    execute_harness,
    validate_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialized-dir", type=Path)
    args = parser.parse_args()

    config = ScaleConfig.from_path(args.config)
    first = execute_harness(config, args.materialized_dir)
    second = execute_harness(config, args.materialized_dir)
    validate_evidence(config, first)
    validate_evidence(config, second)
    comparison = compare_reproducibility(first, second)
    if not comparison["exact_reproduction"]:
        raise SystemExit("deterministic evidence did not reproduce exactly")

    bundle = {
        "bundle_schema_version": "atlasrag-scale-evidence-bundle/v1",
        "config": config.canonical_mapping(),
        "config_sha256": config.sha256,
        "first_run": first.to_mapping(),
        "second_run": second.to_mapping(),
        "reproducibility": comparison,
        "claim_boundary": {
            "allowed": [
                (
                    "The configured smoke corpus and workload were generated "
                    "deterministically."
                ),
                "Shard counts and content digests reproduced exactly across two runs.",
                (
                    "The harness separates executed results from unexecuted target "
                    "configurations."
                ),
            ],
            "forbidden": [
                "Production traffic or service-level objectives.",
                "Distributed retrieval or multi-node serving performance.",
                "A 100M-document execution unless a measured-scale bundle proves it.",
                "Security certification or proof of non-interference.",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
