"""Reciprocal Rank Fusion over lexical and exact dense retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from atlasrag.models import Chunk
from atlasrag.retrieval.bm25 import BM25Retriever
from atlasrag.retrieval.contracts import (
    AccessPrincipal,
    RetrievalContribution,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    ScoreKind,
    resolve_query,
)
from atlasrag.retrieval.dense import ExactDenseRetriever


@dataclass(slots=True)
class _FusedCandidate:
    chunk: Chunk
    score: float
    contributions: list[RetrievalContribution]


class ReciprocalRankFusionRetriever(Retriever):
    """Fuse BM25 and exact dense ranks without calibrating raw scores."""

    def __init__(
        self,
        lexical: BM25Retriever,
        dense: ExactDenseRetriever,
        *,
        rrf_k: int = 60,
        candidate_k: int = 50,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self._lexical = lexical
        self._dense = dense
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    @property
    def method(self) -> RetrievalMethod:
        return RetrievalMethod.HYBRID_RRF

    @property
    def rrf_k(self) -> int:
        return self._rrf_k

    @property
    def candidate_k(self) -> int:
        return self._candidate_k

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Index the same frozen chunk sequence in both component retrievers."""

        chunk_tuple = tuple(chunks)
        self._lexical.index(chunk_tuple)
        self._dense.index(chunk_tuple)

    def search(
        self,
        query: str | RetrievalQuery,
        *,
        top_k: int | None = None,
        principal: AccessPrincipal | None = None,
    ) -> tuple[RetrievalResult, ...]:
        request = resolve_query(query, top_k=top_k, principal=principal)
        component_top_k = max(request.top_k, self._candidate_k)
        component_query = RetrievalQuery(
            text=request.text,
            top_k=component_top_k,
            principal=request.principal,
            excluded_chunk_ids=request.excluded_chunk_ids,
        )
        lexical_results = self._lexical.search(component_query)
        dense_results = self._dense.search(component_query)

        fused: dict[str, _FusedCandidate] = {}
        for result in (*lexical_results, *dense_results):
            candidate = fused.get(result.chunk.chunk_id)
            contribution = RetrievalContribution(
                method=result.method,
                rank=result.rank,
                score=result.score,
                score_kind=result.score_kind,
            )
            if candidate is None:
                fused[result.chunk.chunk_id] = _FusedCandidate(
                    chunk=result.chunk,
                    score=1.0 / (self._rrf_k + result.rank),
                    contributions=[contribution],
                )
                continue
            if candidate.chunk != result.chunk:
                raise ValueError(
                    "component retrievers returned conflicting chunks for one chunk ID"
                )
            candidate.score += 1.0 / (self._rrf_k + result.rank)
            candidate.contributions.append(contribution)

        ordered = sorted(
            fused.values(),
            key=lambda candidate: (-candidate.score, candidate.chunk.chunk_id),
        )
        return tuple(
            RetrievalResult(
                chunk=candidate.chunk,
                score=candidate.score,
                rank=rank,
                method=self.method,
                score_kind=ScoreKind.RECIPROCAL_RANK_FUSION,
                contributions=tuple(
                    sorted(
                        candidate.contributions,
                        key=lambda contribution: contribution.method.value,
                    )
                ),
            )
            for rank, candidate in enumerate(
                ordered[: request.top_k],
                start=1,
            )
        )
