from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "README.md",
    "METHODOLOGY.md",
    "RESULTS.md",
    "LIMITATIONS.md",
    "CLAIM_LEDGER.md",
    "DATASET_PROVENANCE.json",
    "EXPERIMENT_MANIFEST.json",
    "SOURCE_EQUIVALENCE.json",
    "SHA256SUMS",
    "artifacts/installed-package-summary.json",
    "artifacts/installed-package-reproducibility.json",
    "artifacts/atlasrag-adapter-smoke.json",
    "artifacts/regression-gate-report.json",
]
FORBIDDEN = [
    re.compile(r"/home/" + r"tandi", re.IGNORECASE),
    re.compile(r"job" + r"-search", re.IGNORECASE),
    re.compile(r"Gaum" + r"ard", re.IGNORECASE),
    re.compile(r"Get Covered" + r" New Jersey", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
PUBLIC_SUFFIXES = {".md", ".json", ".csv", ".py", ".txt"}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(2)


def load_json(relative_path: str) -> object:
    return json.loads((ROOT / relative_path).read_text(encoding="ascii"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksum_manifest() -> int:
    manifest = ROOT / "SHA256SUMS"
    checked = 0
    for line_number, line in enumerate(
        manifest.read_text(encoding="ascii").splitlines(), start=1
    ):
        if not line:
            continue
        try:
            expected, relative_path = line.split("  ", 1)
        except ValueError:
            fail(f"malformed SHA256SUMS line {line_number}")
        path = ROOT / relative_path
        if not path.is_file():
            fail(f"checksum target is missing: {relative_path}")
        if sha256(path) != expected:
            fail(f"checksum mismatch: {relative_path}")
        checked += 1
    if checked == 0:
        fail("SHA256SUMS contains no entries")
    return checked


for relative_path in REQUIRED:
    if not (ROOT / relative_path).is_file():
        fail(f"missing required artifact: {relative_path}")

for path in ROOT.rglob("*"):
    if path.resolve() == Path(__file__).resolve():
        continue
    if not path.is_file() or path.suffix not in PUBLIC_SUFFIXES:
        continue
    raw = path.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        fail(f"non-ASCII public artifact: {path.relative_to(ROOT)}")
    for pattern in FORBIDDEN:
        if pattern.search(text):
            fail(f"forbidden private/credential pattern in {path.relative_to(ROOT)}")
    if path.suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON {path.relative_to(ROOT)}: {exc}")

source_equivalence = load_json("SOURCE_EQUIVALENCE.json")
assert isinstance(source_equivalence, dict)
if not source_equivalence["runtime_source_equal"]:
    fail("benchmark-to-release runtime source equivalence is false")

reproducibility = load_json("artifacts/installed-package-reproducibility.json")
assert isinstance(reproducibility, dict)
if not reproducibility["passed"]:
    fail("installed-package reproducibility gate failed")

adapter_smoke = load_json("artifacts/atlasrag-adapter-smoke.json")
assert isinstance(adapter_smoke, dict)
if not adapter_smoke["gates"]["passed"]:
    fail("authorization/provenance adapter smoke failed")

regression_gates = load_json("artifacts/regression-gate-report.json")
assert isinstance(regression_gates, dict)
if not regression_gates["passed"]:
    fail("benchmark regression gates failed")

checksum_count = verify_checksum_manifest()
subprocess.run(
    [sys.executable, str(ROOT / "reranking/verify_artifacts.py")],
    check=True,
)
subprocess.run(
    [sys.executable, str(ROOT / "promotion/verify_artifacts.py")],
    check=True,
)
print(
    "PASS: benchmark documentation, JSON, privacy, source-equivalence, "
    "regression gates, reranking and promotion development evidence, "
    f"and {checksum_count} checksums"
)
