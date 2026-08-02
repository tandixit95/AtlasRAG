from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_BENCHMARKS = ROOT.parent
REQUIRED = (
    "README.md",
    "METHODOLOGY.md",
    "RESULTS.md",
    "LIMITATIONS.md",
    "CLAIM_LEDGER.md",
    "MANIFEST.json",
    "SHA256SUMS",
    "artifacts/run-a.json",
    "artifacts/run-b.json",
    "artifacts/quality-summary.csv",
    "artifacts/reproducibility.json",
    "artifacts/rankings.jsonl.gz",
)
FORBIDDEN = (
    re.compile(r"/home/" + r"tandi", re.IGNORECASE),
    re.compile(r"job" + r"-search", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
FORBIDDEN_TEXT_KEYS = {"text", "query_text", "corpus_text", "document_text"}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        fail(f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def contains_forbidden_text_key(value: Any) -> bool:
    if isinstance(value, dict):
        if FORBIDDEN_TEXT_KEYS & set(value):
            return True
        return any(contains_forbidden_text_key(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_forbidden_text_key(item) for item in value)
    return False


for relative in REQUIRED:
    if not (ROOT / relative).is_file():
        fail(f"missing required file: {relative}")

for path in ROOT.rglob("*"):
    if not path.is_file() or path.suffix not in {".md", ".json", ".csv", ".py"}:
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
            json.loads(text)
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON {path.relative_to(ROOT)}: {exc}")

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
    target = ROOT / relative
    if not target.is_file():
        fail(f"missing checksum target: {relative}")
    if sha256(target) != expected:
        fail(f"checksum mismatch: {relative}")
    checked += 1

manifest = load_json(ROOT / "MANIFEST.json")
reproducibility = load_json(ROOT / "artifacts/reproducibility.json")
run_a = load_json(ROOT / "artifacts/run-a.json")
run_b = load_json(ROOT / "artifacts/run-b.json")

if manifest["package"]["git_commit"] != "43c4ef33b212869c94ff8cd9bb1c8615b0084b24":
    fail("unexpected evaluated commit")
if manifest["package"]["wheel_sha256"] != run_a["package"]["wheel_sha256"]:
    fail("manifest/wheel identity mismatch")
if run_a["quality"] != run_b["quality"]:
    fail("A/B quality mismatch")
if not reproducibility["passed"]:
    fail("reproducibility gate failed")
if reproducibility["recommendation"]["enable_by_default"]:
    fail("evidence unexpectedly enables reranking by default")
if reproducibility["recommendation"]["publish_latency_as_stable_component_claim"]:
    fail("evidence unexpectedly approves stable latency publication")
if reproducibility["dominated_candidate_depths"] != {"20": 10}:
    fail("dominated-depth decision mismatch")

for source in manifest["source"].values():
    path = (ROOT / source["path"]).resolve()
    if not path.is_file() or sha256(path) != source["sha256"]:
        fail(f"benchmark source hash mismatch: {source['path']}")

compressed = ROOT / manifest["canonical_rankings"]["compressed_path"]
with gzip.open(compressed, "rt", encoding="ascii") as stream:
    line_count = 0
    digest = hashlib.sha256()
    for line in stream:
        if not line.strip():
            continue
        digest.update(line.encode("ascii"))
        value = json.loads(line)
        if contains_forbidden_text_key(value):
            fail("compressed ranking artifact contains redistributed text")
        line_count += 1
if line_count != manifest["canonical_rankings"]["line_count"]:
    fail("compressed ranking line count mismatch")
if digest.hexdigest() != manifest["canonical_rankings"]["uncompressed_sha256"]:
    fail("compressed ranking uncompressed checksum mismatch")
if line_count != 300:
    fail("unexpected ranking query count")

print(
    "PASS: reranking evidence, package/model identity, privacy boundary, "
    f"reproducibility decision, {line_count} rankings, and {checked} checksums"
)
