"""Deterministic synthetic corpus and workload evidence for AtlasRAG.

This module measures only what it executes. A configuration may describe a future
scale target, but no result is claimable unless ``execution_status`` is
``measured_smoke`` or ``measured_scale`` and every declared count and digest is
present and internally consistent.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "atlasrag-scale-evidence/v1"
RESULT_SCHEMA_VERSION = "atlasrag-scale-result/v1"
EXECUTED_STATUSES = frozenset({"measured_smoke", "measured_scale"})


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _stable_int(*parts: str) -> int:
    payload = "\x1f".join(parts).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class ScaleConfig:
    """Validated input to the deterministic synthetic evidence generator."""

    schema_version: str
    run_id: str
    execution_status: Literal["target_unexecuted", "measured_smoke", "measured_scale"]
    seed: str
    document_count: int
    shard_count: int
    tenant_count: int
    groups_per_tenant: int
    public_fraction: float
    min_text_bytes: int
    max_text_bytes: int
    query_count: int
    update_fraction: float
    materialize_corpus: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if not self.run_id.strip() or not self.seed.strip():
            raise ValueError("run_id and seed must not be blank")
        if self.execution_status not in {
            "target_unexecuted",
            "measured_smoke",
            "measured_scale",
        }:
            raise ValueError("unsupported execution_status")
        if self.document_count <= 0:
            raise ValueError("document_count must be positive")
        if not 1 <= self.shard_count <= self.document_count:
            raise ValueError("shard_count must be between 1 and document_count")
        if self.tenant_count <= 0 or self.groups_per_tenant <= 0:
            raise ValueError("tenant and group counts must be positive")
        if not 0.0 <= self.public_fraction <= 1.0:
            raise ValueError("public_fraction must be between 0 and 1")
        if self.min_text_bytes <= 0 or self.max_text_bytes < self.min_text_bytes:
            raise ValueError("text byte bounds are invalid")
        if self.query_count <= 0:
            raise ValueError("query_count must be positive")
        if not 0.0 <= self.update_fraction <= 1.0:
            raise ValueError("update_fraction must be between 0 and 1")
        if self.execution_status == "target_unexecuted" and self.materialize_corpus:
            raise ValueError("an unexecuted target cannot materialize a corpus")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ScaleConfig:
        return cls(**value)

    @classmethod
    def from_path(cls, path: Path) -> ScaleConfig:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def canonical_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.canonical_mapping()))


@dataclass(frozen=True, slots=True)
class GeneratedDocument:
    """One deterministic synthetic record before optional materialization."""

    document_id: str
    source_uri: str
    text: str
    tenant_id: str | None
    groups: tuple[str, ...]
    shard_id: int

    def canonical_mapping(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "groups": list(self.groups),
            "shard_id": self.shard_id,
            "source_uri": self.source_uri,
            "tenant_id": self.tenant_id,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class ScaleEvidence:
    """Content-addressed evidence returned by one measured harness execution."""

    result_schema_version: str
    execution_status: str
    run_id: str
    config_sha256: str
    document_count: int
    logical_corpus_bytes: int
    shard_count: int
    shard_document_counts: tuple[int, ...]
    shard_sha256: tuple[str, ...]
    corpus_stream_sha256: str
    query_count: int
    query_workload_sha256: str
    update_count: int
    update_workload_sha256: str
    materialized_corpus: bool
    materialized_path: str | None
    materialized_file_sha256: str | None
    elapsed_seconds_diagnostic: float
    environment: dict[str, str]

    def canonical_reproducibility_surface(self) -> dict[str, Any]:
        """Return fields that must reproduce exactly across equivalent runs."""

        value = asdict(self)
        value.pop("elapsed_seconds_diagnostic")
        value.pop("environment")
        value.pop("materialized_path")
        return value

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["shard_document_counts"] = list(self.shard_document_counts)
        value["shard_sha256"] = list(self.shard_sha256)
        return value


_TOPICS = (
    "authorization",
    "retrieval",
    "provenance",
    "latency",
    "evaluation",
    "reranking",
    "ingestion",
    "grounding",
    "reliability",
    "observability",
)


def generate_document(config: ScaleConfig, ordinal: int) -> GeneratedDocument:
    """Generate one exact synthetic record without using process-global randomness."""

    if not 0 <= ordinal < config.document_count:
        raise IndexError("document ordinal is outside configured corpus")

    digest_int = _stable_int(config.seed, "document", str(ordinal))
    document_id = f"doc-{ordinal:012d}"
    shard_id = digest_int % config.shard_count
    is_public = (digest_int % 1_000_000) < int(config.public_fraction * 1_000_000)
    tenant_number = digest_int % config.tenant_count
    group_number = (
        digest_int // max(config.tenant_count, 1)
    ) % config.groups_per_tenant
    tenant_id = None if is_public else f"tenant-{tenant_number:04d}"
    groups = () if is_public else (f"group-{group_number:03d}",)
    topic = _TOPICS[(digest_int // 17) % len(_TOPICS)]
    unique_term = f"needle_{ordinal:012d}"
    target_bytes = config.min_text_bytes + (
        digest_int % (config.max_text_bytes - config.min_text_bytes + 1)
    )
    prefix = (
        f"Synthetic AtlasRAG evidence record {document_id}. "
        f"Topic {topic}. Unique token {unique_term}. "
        "No employer, customer, or private corpus material is present. "
    )
    filler = (
        "retrieval provenance authorization evaluation reliability grounding "
        "deterministic evidence "
    )
    text = prefix
    while len(text.encode("utf-8")) < target_bytes:
        text += filler
    encoded = text.encode("utf-8")[:target_bytes]
    text = encoded.decode("utf-8", errors="strict")
    return GeneratedDocument(
        document_id=document_id,
        source_uri=f"synthetic://atlasrag-scale/{document_id}",
        text=text,
        tenant_id=tenant_id,
        groups=groups,
        shard_id=shard_id,
    )


def _query_record(config: ScaleConfig, ordinal: int) -> dict[str, Any]:
    target = _stable_int(config.seed, "query", str(ordinal)) % config.document_count
    document = generate_document(config, target)
    return {
        "query_id": f"query-{ordinal:010d}",
        "query": f"needle_{target:012d}",
        "expected_document_id": document.document_id,
        "principal": {
            "groups": list(document.groups),
            "tenant_id": document.tenant_id,
        },
    }


def _update_record(config: ScaleConfig, ordinal: int) -> dict[str, Any]:
    target = _stable_int(config.seed, "update", str(ordinal)) % config.document_count
    return {
        "update_id": f"update-{ordinal:010d}",
        "document_id": f"doc-{target:012d}",
        "version": 2,
    }


def execute_harness(
    config: ScaleConfig, output_dir: Path | None = None
) -> ScaleEvidence:
    """Execute one measured synthetic generation and integrity run.

    ``target_unexecuted`` configurations are deliberately refused. This protects
    target sizes from being mistaken for measured results.
    """

    if config.execution_status not in EXECUTED_STATUSES:
        raise ValueError("target_unexecuted configurations cannot produce results")
    if config.materialize_corpus and output_dir is None:
        raise ValueError("output_dir is required when materialize_corpus is true")

    started = time.perf_counter()
    corpus_hasher = sha256()
    shard_hashers = [sha256() for _ in range(config.shard_count)]
    shard_counts = [0] * config.shard_count
    logical_bytes = 0
    materialized_path: Path | None = None
    materialized_handle = None
    try:
        if config.materialize_corpus:
            assert output_dir is not None
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            materialized_path = output_dir / "synthetic-corpus.jsonl"
            materialized_handle = materialized_path.open("wb")
        for ordinal in range(config.document_count):
            document = generate_document(config, ordinal)
            record = _canonical_json_bytes(document.canonical_mapping()) + b"\n"
            corpus_hasher.update(record)
            shard_hashers[document.shard_id].update(record)
            shard_counts[document.shard_id] += 1
            logical_bytes += len(document.text.encode("utf-8"))
            if materialized_handle is not None:
                materialized_handle.write(record)
    finally:
        if materialized_handle is not None:
            materialized_handle.close()

    query_hasher = sha256()
    for ordinal in range(config.query_count):
        query_hasher.update(
            _canonical_json_bytes(_query_record(config, ordinal)) + b"\n"
        )

    update_count = int(config.document_count * config.update_fraction)
    update_hasher = sha256()
    for ordinal in range(update_count):
        update_hasher.update(
            _canonical_json_bytes(_update_record(config, ordinal)) + b"\n"
        )

    materialized_sha = None
    if materialized_path is not None:
        materialized_sha = _sha256_bytes(materialized_path.read_bytes())
        if materialized_sha != corpus_hasher.hexdigest():
            raise RuntimeError(
                "materialized corpus digest does not match stream digest"
            )

    elapsed = time.perf_counter() - started
    return ScaleEvidence(
        result_schema_version=RESULT_SCHEMA_VERSION,
        execution_status=config.execution_status,
        run_id=config.run_id,
        config_sha256=config.sha256,
        document_count=config.document_count,
        logical_corpus_bytes=logical_bytes,
        shard_count=config.shard_count,
        shard_document_counts=tuple(shard_counts),
        shard_sha256=tuple(hasher.hexdigest() for hasher in shard_hashers),
        corpus_stream_sha256=corpus_hasher.hexdigest(),
        query_count=config.query_count,
        query_workload_sha256=query_hasher.hexdigest(),
        update_count=update_count,
        update_workload_sha256=update_hasher.hexdigest(),
        materialized_corpus=config.materialize_corpus,
        materialized_path=None if materialized_path is None else str(materialized_path),
        materialized_file_sha256=materialized_sha,
        elapsed_seconds_diagnostic=round(elapsed, 6),
        environment={
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
    )


def validate_evidence(config: ScaleConfig, evidence: ScaleEvidence) -> None:
    """Fail closed when a result cannot support its stated execution claim."""

    if config.execution_status not in EXECUTED_STATUSES:
        raise ValueError(
            "unexecuted target configurations cannot have claimable evidence"
        )
    if evidence.execution_status != config.execution_status:
        raise ValueError("result execution_status does not match configuration")
    if evidence.result_schema_version != RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported result schema")
    if evidence.config_sha256 != config.sha256:
        raise ValueError("result config digest does not match configuration")
    if evidence.document_count != config.document_count:
        raise ValueError("result document count does not match configuration")
    if evidence.shard_count != config.shard_count:
        raise ValueError("result shard count does not match configuration")
    if len(evidence.shard_document_counts) != config.shard_count:
        raise ValueError("shard count vector has wrong length")
    if len(evidence.shard_sha256) != config.shard_count:
        raise ValueError("shard digest vector has wrong length")
    if sum(evidence.shard_document_counts) != config.document_count:
        raise ValueError("shard document counts do not sum to corpus size")
    if evidence.query_count != config.query_count:
        raise ValueError("query count does not match configuration")
    expected_updates = int(config.document_count * config.update_fraction)
    if evidence.update_count != expected_updates:
        raise ValueError("update count does not match configuration")
    digests = (
        evidence.corpus_stream_sha256,
        evidence.query_workload_sha256,
        evidence.update_workload_sha256,
        *evidence.shard_sha256,
    )
    if any(len(value) != 64 for value in digests):
        raise ValueError("one or more evidence digests are not SHA-256 values")
    if evidence.materialized_corpus != config.materialize_corpus:
        raise ValueError("materialization state does not match configuration")
    if evidence.materialized_corpus and evidence.materialized_file_sha256 is None:
        raise ValueError("materialized evidence is missing its file digest")


def compare_reproducibility(
    first: ScaleEvidence, second: ScaleEvidence
) -> dict[str, Any]:
    """Compare exact deterministic evidence while excluding diagnostic timing."""

    first_surface = first.canonical_reproducibility_surface()
    second_surface = second.canonical_reproducibility_surface()
    return {
        "comparison_schema_version": "atlasrag-scale-reproducibility/v1",
        "exact_reproduction": first_surface == second_surface,
        "first_surface_sha256": _sha256_bytes(_canonical_json_bytes(first_surface)),
        "second_surface_sha256": _sha256_bytes(_canonical_json_bytes(second_surface)),
    }


def evidence_from_mapping(value: dict[str, Any]) -> ScaleEvidence:
    value = dict(value)
    value["shard_document_counts"] = tuple(value["shard_document_counts"])
    value["shard_sha256"] = tuple(value["shard_sha256"])
    return ScaleEvidence(**value)
