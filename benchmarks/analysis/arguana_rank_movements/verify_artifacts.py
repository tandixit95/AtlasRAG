from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_IMPORT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_IMPORT_ROOT))

from benchmarks.src.analyze_promotion_rank_movements import (  # noqa: E402
    analyze_rows,
    contains_forbidden_text_key,
    load_rows,
    sha256,
)

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
SOURCE = REPO_ROOT / "benchmarks/promotion/artifacts/arguana-rankings.jsonl.gz"
EXPECTED_SOURCE_SHA256 = (
    "32dab7b69e57cf98d9fc96dd09016cb26d2b6bcce17fff50fa1fb192ac143bc5"
)
REQUIRED = (
    "README.md",
    "METHODOLOGY.md",
    "RESULTS.md",
    "LIMITATIONS.md",
    "CLAIM_LEDGER.md",
    "MANIFEST.json",
    "SHA256SUMS",
    "artifacts/analysis.json",
)
FORBIDDEN = (
    re.compile(r"/home/" + r"tandi", re.IGNORECASE),
    re.compile(r"job" + r"-search", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(2)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        fail(f"expected JSON object: {path}")
    return value


for relative in REQUIRED:
    if not (ROOT / relative).is_file():
        fail(f"missing required file: {relative}")

for path in ROOT.rglob("*"):
    if not path.is_file() or path.suffix not in {".md", ".json", ".py"}:
        continue
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeDecodeError:
        fail(f"non-ASCII public artifact: {path.relative_to(ROOT)}")
    for pattern in FORBIDDEN:
        if pattern.search(text):
            fail(f"forbidden private or credential pattern: {path.relative_to(ROOT)}")
    if path.suffix == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        if contains_forbidden_text_key(value):
            fail(f"forbidden payload text key: {path.relative_to(ROOT)}")

checked = 0
for line_number, line in enumerate(
    (ROOT / "SHA256SUMS").read_text(encoding="ascii").splitlines(), start=1
):
    if not line:
        continue
    try:
        expected, relative = line.split("  ", 1)
    except ValueError:
        fail(f"malformed SHA256SUMS line {line_number}")
    target = (ROOT / relative).resolve()
    if not target.is_file():
        fail(f"missing checksum target: {relative}")
    if file_sha256(target) != expected:
        fail(f"checksum mismatch: {relative}")
    checked += 1

manifest = load_json(ROOT / "MANIFEST.json")
if manifest.get("schema_version") != "atlasrag.arguana-rank-analysis-manifest.v1":
    fail("unexpected manifest schema")
if manifest["source"]["sha256"] != EXPECTED_SOURCE_SHA256:
    fail("manifest source hash changed")
if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
    fail("frozen source ranking hash changed")
for entry in manifest["files"]:
    target = (ROOT / entry["path"]).resolve()
    if not target.is_file() or file_sha256(target) != entry["sha256"]:
        fail(f"manifest file hash mismatch: {entry['path']}")

artifact = load_json(ROOT / "artifacts/analysis.json")
if artifact.get("schema_version") != "atlasrag.arguana-rank-movement-analysis.v1":
    fail("unexpected analysis schema")
if artifact["source"]["sha256"] != EXPECTED_SOURCE_SHA256:
    fail("artifact source hash changed")
if artifact["source"]["query_count"] != 200:
    fail("unexpected query count")
if artifact["source"]["candidate_set_preserved_queries"] != 200:
    fail("candidate set preservation changed")
if artifact["classification_counts"] != {
    "both_absent": 31,
    "improved": 45,
    "lost": 0,
    "regressed": 80,
    "rescued": 0,
    "unchanged": 44,
}:
    fail("rank movement counts changed")
if artifact["metric_movements"]["mrr@10"]["mean_delta"] != -0.06093452381:
    fail("MRR delta changed")
if artifact["metric_movements"]["ndcg@10"]["mean_delta"] != -0.04764053288:
    fail("nDCG delta changed")
if artifact["metric_movements"]["recall@10"]["mean_delta"] != 0.0:
    fail("recall delta changed")
if artifact["rank_movements"]["mean_rank_delta"] != 0.579881656805:
    fail("mean rank delta changed")
if artifact["interpretation"]["candidate_pool_changed"]:
    fail("artifact unexpectedly reports candidate-pool change")
if artifact["interpretation"]["candidate_recall_changed"]:
    fail("artifact unexpectedly reports recall change")

rows = load_rows(SOURCE)
regenerated = analyze_rows(
    rows,
    source_path="benchmarks/promotion/artifacts/arguana-rankings.jsonl.gz",
    source_sha256=EXPECTED_SOURCE_SHA256,
)
if regenerated != artifact:
    fail("analysis artifact does not reproduce exactly")

print(
    "PASS: no-payload ArguAna rank-movement analysis, exact regeneration, "
    f"and {checked} checksums"
)
