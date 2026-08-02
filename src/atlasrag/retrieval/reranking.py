"""Reranking contracts, adapters, and retrieval composition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import isfinite

from atlasrag.models import Chunk
from atlasrag.retrieval.contracts import (
    AccessPrincipal,
    RerankTrace,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    ScoreKind,
    resolve_query,
)


class Reranker(ABC):
    """Model-independent boundary for scoring query/chunk pairs."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable model or implementation identifier."""

    @abstractmethod
    def score(self, query: str, chunks: Sequence[Chunk]) -> Sequence[float]:
        """Return exactly one relevance score per chunk, in input order."""


class CrossEncoderReranker(Reranker):
    """Optional Sentence Transformers cross-encoder adapter.

    The model dependency is imported lazily. AtlasRAG core and deterministic
    contract tests therefore remain standard-library-only.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        *,
        revision: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "CrossEncoderReranker requires the 'reranking' extra"
            ) from exc

        self._model_name = model_name.strip()
        self._revision = None if revision is None else revision.strip()
        if self._revision == "":
            raise ValueError("revision must not be blank")
        self._batch_size = batch_size
        self._model = CrossEncoder(
            self._model_name,
            revision=self._revision,
            device=device,
            local_files_only=local_files_only,
        )

    @property
    def model_id(self) -> str:
        if self._revision is None:
            return self._model_name
        return f"{self._model_name}@{self._revision}"

    @property
    def revision(self) -> str | None:
        return self._revision

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def score(self, query: str, chunks: Sequence[Chunk]) -> tuple[float, ...]:
        if not query.strip():
            raise ValueError("query text must not be blank")
        if not chunks:
            return ()
        pairs = [(query, chunk.text) for chunk in chunks]
        raw_scores = self._model.predict(pairs, batch_size=self._batch_size)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        return tuple(float(score) for score in raw_scores)


class RerankedRetriever(Retriever):
    """Rerank an authorization-safe candidate set from another retriever.

    The wrapped retriever remains responsible for indexing, authorization, and
    exclusions. Reranking sees only returned candidates and cannot introduce a
    chunk that the candidate stage did not authorize.
    """

    def __init__(
        self,
        candidate_retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_k: int = 50,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self._candidate_retriever = candidate_retriever
        self._reranker = reranker
        self._candidate_k = candidate_k

    @property
    def method(self) -> RetrievalMethod:
        return RetrievalMethod.RERANKED

    @property
    def candidate_k(self) -> int:
        return self._candidate_k

    @property
    def model_id(self) -> str:
        return self._reranker.model_id

    @property
    def candidate_method(self) -> RetrievalMethod:
        return self._candidate_retriever.method

    def search(
        self,
        query: str | RetrievalQuery,
        *,
        top_k: int | None = None,
        principal: AccessPrincipal | None = None,
    ) -> tuple[RetrievalResult, ...]:
        request = resolve_query(query, top_k=top_k, principal=principal)
        candidate_query = RetrievalQuery(
            text=request.text,
            top_k=max(request.top_k, self._candidate_k),
            principal=request.principal,
            excluded_chunk_ids=request.excluded_chunk_ids,
        )
        candidates = self._candidate_retriever.search(candidate_query)
        if not candidates:
            return ()

        chunk_ids = [candidate.chunk.chunk_id for candidate in candidates]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("candidate retriever returned duplicate chunk IDs")

        scores = tuple(
            float(score)
            for score in self._reranker.score(
                request.text,
                [candidate.chunk for candidate in candidates],
            )
        )
        if len(scores) != len(candidates):
            raise ValueError("reranker must return exactly one score per candidate")
        if any(not isfinite(score) for score in scores):
            raise ValueError("reranker scores must be finite")

        scored = list(zip(candidates, scores, strict=True))
        scored.sort(
            key=lambda item: (
                -item[1],
                item[0].rank,
                item[0].chunk.chunk_id,
            )
        )
        return tuple(
            RetrievalResult(
                chunk=candidate.chunk,
                score=score,
                rank=rank,
                method=self.method,
                score_kind=ScoreKind.RERANKER_RELEVANCE,
                contributions=candidate.contributions,
                rerank_trace=RerankTrace(
                    model_id=self.model_id,
                    candidate_method=candidate.method,
                    candidate_score_kind=candidate.score_kind,
                    candidate_score=candidate.score,
                    candidate_rank=candidate.rank,
                    candidate_contributions=candidate.contributions,
                ),
            )
            for rank, (candidate, score) in enumerate(
                scored[: request.top_k],
                start=1,
            )
        )
