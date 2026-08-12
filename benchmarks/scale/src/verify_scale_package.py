#!/usr/bin/env python3
"""Verify the complete committed scale-evidence package and its checksums."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from benchmarks.scale.src.scale_harness import ScaleConfig
from benchmarks.scale.src.verify_disk_benchmark_protocol import load_and_validate
from benchmarks.scale.src.verify_scale_evidence import main as verify_bundle_main

ROOT = Path(__file__).resolve().parents[3]
SCALE_ROOT = ROOT / "benchmarks" / "scale"


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest_path = SCALE_ROOT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "atlasrag-scale-package-manifest/v1":
        raise SystemExit("unsupported scale package manifest")

    errors: list[str] = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            errors.append(f"missing: {entry['path']}")
            continue
        actual = file_sha256(path)
        if actual != entry["sha256"]:
            errors.append(
                f"digest mismatch: {entry['path']} expected={entry['sha256']} "
                f"actual={actual}"
            )

    disk_protocol = load_and_validate(
        SCALE_ROOT / "protocols" / "disk-backed-v1-unexecuted.json"
    )
    if disk_protocol["execution_status"] != "protocol_unexecuted":
        errors.append("disk benchmark protocol must remain protocol_unexecuted")

    target = ScaleConfig.from_path(
        SCALE_ROOT / "configs" / "target-100m-unexecuted.json"
    )
    if target.execution_status != "target_unexecuted":
        errors.append("100M target must remain target_unexecuted")

    if errors:
        raise SystemExit("\n".join(errors))

    # Reuse the bundle verifier through its normal CLI contract.
    import sys

    previous = sys.argv
    try:
        sys.argv = [
            "verify_scale_evidence",
            str(SCALE_ROOT / "artifacts" / "smoke-v1.json"),
        ]
        verify_bundle_main()
    finally:
        sys.argv = previous

    print(
        json.dumps(
            {
                "files_verified": len(manifest["files"]),
                "status": "PASS",
                "target_100m_status": target.execution_status,
                "disk_protocol_status": disk_protocol["execution_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
