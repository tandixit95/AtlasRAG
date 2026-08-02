"""Typed query and result contracts shared by retrieval implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from types import MappingProxyType

from atlasrag.models import Chunk


class RetrievalMethod(StrEnum):
    """Stable identifiers for AtlasRAG retrieval implementations."""

    EXACT_DENSE = "exact_dense"
    BM25 = "bm25"
    HYBRID_RRF = "hybrid_rrf"
    RERANKED = "reranked"


class ScoreKind(StrEnum):
    """Score semantics, kept explicit because methods are not calibrated."""

    COSINE_SIMILARITY = "cosine_similarity"
    BM25 = "bm25"
    RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"
    RERANKER_RELEVANCE = "reranker_relevance"


def _normalize_nonblank(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _validate_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a SHA-256 hex digest") from exc


def _normalize_groups(groups: Iterable[str]) -> frozenset[str]:
    if isinstance(groups, str):
        raise TypeError("groups must be an iterable of group names, not a string")
    normalized: set[str] = set()
    for group in groups:
        value = _normalize_nonblank(group, field_name="group")
        if "," in value:
            raise ValueError("group names must not contain commas")
        normalized.add(value)
    return frozenset(normalized)


def _normalize_chunk_ids(chunk_ids: Iterable[str]) -> frozenset[str]:
    if isinstance(chunk_ids, str):
        raise TypeError(
            "excluded_chunk_ids must be an iterable of chunk IDs, not a string"
        )
    return frozenset(
        _normalize_nonblank(chunk_id, field_name="excluded chunk ID")
        for chunk_id in chunk_ids
    )


@dataclass(frozen=True, slots=True)
class AccessPrincipal:
    """Caller identity used for permission-aware retrieval.

    A principal may carry a tenant identity, zero or more groups, or both.
    Public chunks remain visible to every principal.
    """

    tenant_id: str | None = None
    groups: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        tenant_id = self.tenant_id
        if tenant_id is not None:
            tenant_id = _normalize_nonblank(tenant_id, field_name="tenant_id")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "groups", _normalize_groups(self.groups))


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """Validated retrieval request with explicit authorization context."""

    text: str
    top_k: int = 5
    principal: AccessPrincipal = field(default_factory=AccessPrincipal)
    excluded_chunk_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("query text must not be blank")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not isinstance(self.principal, AccessPrincipal):
            raise TypeError("principal must be an AccessPrincipal")
        object.__setattr__(
            self,
            "excluded_chunk_ids",
            _normalize_chunk_ids(self.excluded_chunk_ids),
        )


@dataclass(frozen=True, slots=True)
class RetrievalContribution:
    """One component rank and raw score contributing to a fused result."""

    method: RetrievalMethod
    rank: int
    score: float
    score_kind: ScoreKind

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("contribution rank must be positive")
        if not isfinite(self.score):
            raise ValueError("contribution score must be finite")


@dataclass(frozen=True, slots=True)
class Citation:
    """Stable source-span citation projection for a retrieved chunk."""

    chunk_id: str
    document_id: str
    document_content_sha256: str
    source_uri: str
    start_char: int
    end_char: int
    content_sha256: str
    strategy_id: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("citation chunk_id must not be blank")
        if not self.document_id.strip():
            raise ValueError("citation document_id must not be blank")
        if not self.source_uri.strip():
            raise ValueError("citation source_uri must not be blank")
        _validate_sha256(
            self.document_content_sha256,
            field_name="citation document_content_sha256",
        )
        _validate_sha256(self.content_sha256, field_name="citation content_sha256")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError("citation span must be a non-empty half-open range")
        if not self.strategy_id.strip():
            raise ValueError("citation strategy_id must not be blank")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> Citation:
        """Project immutable citation fields from ``chunk``."""

        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_content_sha256=chunk.document_content_sha256,
            source_uri=chunk.source_uri,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            content_sha256=chunk.content_sha256,
            strategy_id=chunk.strategy_id,
            metadata=chunk.metadata,
        )


@dataclass(frozen=True, slots=True)
class RerankTrace:
    """Candidate-stage evidence retained after reranking."""

    model_id: str
    candidate_method: RetrievalMethod
    candidate_score_kind: ScoreKind
    candidate_score: float
    candidate_rank: int
    candidate_contributions: tuple[RetrievalContribution, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("rerank model_id must not be blank")
        if self.candidate_rank <= 0:
            raise ValueError("candidate_rank must be positive")
        if not isfinite(self.candidate_score):
            raise ValueError("candidate_score must be finite")
        methods = [contribution.method for contribution in self.candidate_contributions]
        if len(methods) != len(set(methods)):
            raise ValueError(
                "candidate_contributions must contain at most one entry per method"
            )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked result with provenance and explicit score semantics.

    ``score`` is meaningful only within ``method`` and ``score_kind``. Hybrid
    retrieval exposes each component's original rank and raw score separately;
    those raw scores are never treated as calibrated or added together.
    """

    chunk: Chunk
    score: float
    rank: int
    method: RetrievalMethod
    score_kind: ScoreKind
    contributions: tuple[RetrievalContribution, ...] = ()
    rerank_trace: RerankTrace | None = None

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        methods = [contribution.method for contribution in self.contributions]
        if len(methods) != len(set(methods)):
            raise ValueError("contributions must contain at most one entry per method")
        if self.method is RetrievalMethod.RERANKED:
            if self.score_kind is not ScoreKind.RERANKER_RELEVANCE:
                raise ValueError("reranked results require reranker relevance scores")
            if self.rerank_trace is None:
                raise ValueError("reranked results require a rerank trace")
            if self.contributions != self.rerank_trace.candidate_contributions:
                raise ValueError(
                    "reranked contributions must match candidate-stage contributions"
                )
        elif self.rerank_trace is not None:
            raise ValueError("only reranked results may carry a rerank trace")

    @property
    def citation(self) -> Citation:
        """Return an immutable source-span citation for this result."""

        return Citation.from_chunk(self.chunk)


class Retriever(ABC):
    """Framework-independent retrieval boundary."""

    @property
    @abstractmethod
    def method(self) -> RetrievalMethod:
        """Return this retriever's stable method identifier."""

    @abstractmethod
    def search(
        self,
        query: str | RetrievalQuery,
        *,
        top_k: int | None = None,
        principal: AccessPrincipal | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Return ranked, authorization-safe results."""


def resolve_query(
    query: str | RetrievalQuery,
    *,
    top_k: int | None = None,
    principal: AccessPrincipal | None = None,
) -> RetrievalQuery:
    """Normalize the convenience string API into ``RetrievalQuery``."""

    if isinstance(query, RetrievalQuery):
        if top_k is not None or principal is not None:
            raise TypeError(
                "top_k and principal overrides are not allowed with RetrievalQuery"
            )
        return query
    return RetrievalQuery(
        text=query,
        top_k=5 if top_k is None else top_k,
        principal=AccessPrincipal() if principal is None else principal,
    )
