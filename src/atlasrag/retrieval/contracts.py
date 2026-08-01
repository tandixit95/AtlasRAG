"""Typed query and result contracts shared by retrieval implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite

from atlasrag.models import Chunk


class RetrievalMethod(StrEnum):
    """Stable identifiers for AtlasRAG retrieval implementations."""

    EXACT_DENSE = "exact_dense"
    BM25 = "bm25"
    HYBRID_RRF = "hybrid_rrf"


class ScoreKind(StrEnum):
    """Score semantics, kept explicit because methods are not calibrated."""

    COSINE_SIMILARITY = "cosine_similarity"
    BM25 = "bm25"
    RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"


def _normalize_nonblank(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


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

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("query text must not be blank")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not isinstance(self.principal, AccessPrincipal):
            raise TypeError("principal must be an AccessPrincipal")


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

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        methods = [contribution.method for contribution in self.contributions]
        if len(methods) != len(set(methods)):
            raise ValueError("contributions must contain at most one entry per method")


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
