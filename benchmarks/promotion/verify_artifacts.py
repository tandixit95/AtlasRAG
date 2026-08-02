from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
FROZEN_GATE_SHA256 = "ff186c5cd42839478d7b3e7f40377383cec2d7c472b5194eeeb4e70484c217c4"
FROZEN_PROTOCOL_COMMIT = "c59485d698c41797dc307b81fa8a4198f1113812"
REQUIRED = (
    "README.md",
    "METHODOLOGY.md",
    "RESULTS.md",
    "LIMITATIONS.md",
    "CLAIM_LEDGER.md",
    "PROMOTION_GATES.json",
    "MANIFEST.json",
    "SHA256SUMS",
    "artifacts/contract-evaluation.json",
    "artifacts/scifact-run-a.json",
    "artifacts/scifact-run-b.json",
    "artifacts/arguana-run-a.json",
    "artifacts/arguana-run-b.json",
    "artifacts/scifact-summary.csv",
    "artifacts/arguana-summary.csv",
    "artifacts/promotion-report.json",
    "artifacts/scifact-rankings.jsonl.gz",
    "artifacts/arguana-rankings.jsonl.gz",
)
FORBIDDEN = (
    re.compile(r"/home/" + r"tandi", re.IGNORECASE),
    re.compile(r"job" + r"-search", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
FORBIDDEN_TEXT_KEYS = {"text", "query_text", "corpus_text", "document_text"}
EXPECTED_FAILED_GATES = {
    ("scifact-test-300", "reranker_p95_budget_ms"),
    ("scifact-test-300", "primary_metric_interval_excludes_zero"),
    ("arguana-contrast-200", "reranker_p95_budget_ms"),
    ("arguana-contrast-200", "primary_metric_positive_mean"),
    ("arguana-contrast-200", "primary_metric_interval_excludes_zero"),
    ("arguana-contrast-200", "secondary_metric_non_regression"),
}


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

if sha256(ROOT / "PROMOTION_GATES.json") != FROZEN_GATE_SHA256:
    fail("frozen promotion gate hash changed")

manifest = load_json(ROOT / "MANIFEST.json")
report = load_json(ROOT / "artifacts/promotion-report.json")
contracts = load_json(ROOT / "artifacts/contract-evaluation.json")
scifact_a = load_json(ROOT / "artifacts/scifact-run-a.json")
scifact_b = load_json(ROOT / "artifacts/scifact-run-b.json")
arguana_a = load_json(ROOT / "artifacts/arguana-run-a.json")
arguana_b = load_json(ROOT / "artifacts/arguana-run-b.json")

if manifest["protocol"]["frozen_commit"] != FROZEN_PROTOCOL_COMMIT:
    fail("unexpected frozen protocol commit")
if manifest["protocol"]["gates_sha256"] != FROZEN_GATE_SHA256:
    fail("manifest gate hash mismatch")
if manifest["decision"]["disposition"] != "retain_default_rejected":
    fail("manifest decision is not rejected")
if manifest["decision"]["candidate_promoted"]:
    fail("manifest unexpectedly promotes candidate")

if report["candidate_promoted"]:
    fail("report unexpectedly promotes candidate")
decision = report["decision"]
if decision["disposition"] != "retain_default_rejected":
    fail("report disposition mismatch")
if (decision["check_count"], decision["failed_count"], decision["missing_count"]) != (
    37,
    6,
    0,
):
    fail("report gate counts changed")
failed_gates = {
    (check["task_id"], check["gate_id"])
    for check in decision["checks"]
    if check["status"] == "fail"
}
if failed_gates != EXPECTED_FAILED_GATES:
    fail("failed gate set changed")

expected_contracts = {
    "authorization_leakage_count": 0,
    "authorized_private_visible": True,
    "citation_completeness": 1.0,
    "deterministic_ranking_reproduction": True,
    "excluded_chunk_leakage_count": 0,
    "malformed_policy_fails_closed": True,
    "unauthorized_candidate_scoring_count": 0,
}
if contracts["contracts"] != expected_contracts:
    fail("safety contract result changed")

for task_id, run_a, run_b, expected_queries in (
    ("scifact-test-300", scifact_a, scifact_b, 300),
    ("arguana-contrast-200", arguana_a, arguana_b, 200),
):
    if run_a["task_id"] != task_id or run_b["task_id"] != task_id:
        fail(f"task identity mismatch: {task_id}")
    if run_a["dataset"]["query_count"] != expected_queries:
        fail(f"query count mismatch: {task_id}")
    for key in (
        "protocol_id",
        "task_id",
        "dataset",
        "package",
        "configuration",
        "models",
        "quality",
        "environment",
        "limitations",
    ):
        if run_a[key] != run_b[key]:
            fail(f"A/B invariant mismatch: {task_id}.{key}")
    task_evidence = report["evidence"]["tasks"][task_id]
    if not all(task_evidence["reproducibility"].values()):
        fail(f"A/B reproduction failed: {task_id}")
    if task_evidence["privacy"]["payload_redistributed"]:
        fail(f"payload redistribution detected: {task_id}")
    if task_evidence["citations"]["completeness"] != 1.0:
        fail(f"citation completeness failed: {task_id}")
    if not task_evidence["latency"]["controlled_host"]:
        fail(f"host control failed: {task_id}")
    if not task_evidence["latency"]["dispersion_acceptable"]:
        fail(f"latency dispersion failed: {task_id}")

scifact_quality = report["evidence"]["tasks"]["scifact-test-300"]["quality"]
arguana_quality = report["evidence"]["tasks"]["arguana-contrast-200"]["quality"]
if scifact_quality["mrr@10"]["bootstrap_95_percent_interval"][0] >= 0.0:
    fail("SciFact MRR interval unexpectedly excludes zero")
if arguana_quality["mrr@10"]["bootstrap_95_percent_interval"][1] >= 0.0:
    fail("ArguAna MRR interval no longer shows regression")
if arguana_quality["ndcg@10"]["bootstrap_95_percent_interval"][1] >= 0.0:
    fail("ArguAna nDCG interval no longer shows regression")

for source in manifest["source"].values():
    path = (ROOT / source["path"]).resolve()
    if not path.is_file() or sha256(path) != source["sha256"]:
        fail(f"source hash mismatch: {source['path']}")

ranking_count = 0
for task in manifest["tasks"].values():
    ranking = task["rankings"]
    compressed = ROOT / ranking["compressed_path"]
    if sha256(compressed) != ranking["compressed_sha256"]:
        fail(f"compressed ranking hash mismatch: {ranking['compressed_path']}")
    digest = hashlib.sha256()
    lines = 0
    with gzip.open(compressed, "rt", encoding="ascii") as stream:
        for line in stream:
            if not line.strip():
                continue
            digest.update(line.encode("ascii"))
            value = json.loads(line)
            if contains_forbidden_text_key(value):
                fail("compressed ranking artifact contains redistributed text")
            lines += 1
    if lines != ranking["line_count"]:
        fail(f"ranking line count mismatch: {ranking['compressed_path']}")
    if digest.hexdigest() != ranking["uncompressed_sha256"]:
        fail(f"uncompressed ranking hash mismatch: {ranking['compressed_path']}")
    ranking_count += lines

print(
    "PASS: frozen promotion gates, rejected default change, A/B reproduction, "
    f"safety/citations, {ranking_count} rankings, and {checked} checksums"
)
