"""Exact dense retrieval baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt

from atlasrag.embeddings.base import EmbeddingModel, Vector
from atlasrag.models import Chunk
from atlasrag.retrieval.access import PermissionPolicy, policy_for_chunk
from atlasrag.retrieval.contracts import (
    AccessPrincipal,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    ScoreKind,
    resolve_query,
)


@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    chunk: Chunk
    vector: Vector
    policy: PermissionPolicy


def _validate_vector(vector: Vector, *, dimension: int, name: str) -> None:
    if len(vector) != dimension:
        raise ValueError(f"{name} dimension does not match embedder contract")
    if not vector:
        raise ValueError(f"{name} must not be empty")
    if any(not isfinite(value) for value in vector):
        raise ValueError(f"{name} values must be finite")
    if not any(value != 0.0 for value in vector):
        raise ValueError(f"{name} must not be a zero vector")


def _cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions must match")
    if not left:
        raise ValueError("vectors must not be empty")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine similarity is undefined for zero vectors")
    return dot / (left_norm * right_norm)


class ExactDenseRetriever(Retriever):
    """In-memory exhaustive cosine search used as the correctness baseline."""

    def __init__(self, embedder: EmbeddingModel) -> None:
        self._embedder = embedder
        self._indexed: tuple[_IndexedChunk, ...] = ()

    @property
    def method(self) -> RetrievalMethod:
        return RetrievalMethod.EXACT_DENSE

    @property
    def model_id(self) -> str:
        return self._embedder.model_id

    def index(self, chunks: Sequence[Chunk]) -> None:
        chunk_tuple = tuple(chunks)
        chunk_ids = [chunk.chunk_id for chunk in chunk_tuple]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk IDs must be unique within an index")
        policies = tuple(policy_for_chunk(chunk) for chunk in chunk_tuple)
        if not chunk_tuple:
            self._indexed = ()
            return

        vectors = tuple(
            self._embedder.embed_documents([chunk.text for chunk in chunk_tuple])
        )
        if len(vectors) != len(chunk_tuple):
            raise ValueError("embedder must return exactly one vector per chunk")
        for vector in vectors:
            _validate_vector(
                vector,
                dimension=self._embedder.dimension,
                name="embedding",
            )
        self._indexed = tuple(
            _IndexedChunk(chunk=chunk, vector=vector, policy=policy)
            for chunk, vector, policy in zip(
                chunk_tuple,
                vectors,
                policies,
                strict=True,
            )
        )

    def search(
        self,
        query: str | RetrievalQuery,
        *,
        top_k: int | None = None,
        principal: AccessPrincipal | None = None,
    ) -> tuple[RetrievalResult, ...]:
        request = resolve_query(query, top_k=top_k, principal=principal)
        visible = tuple(
            item for item in self._indexed if item.policy.allows(request.principal)
        )
        if not visible:
            return ()

        query_vector = self._embedder.embed_query(request.text)
        _validate_vector(
            query_vector,
            dimension=self._embedder.dimension,
            name="query embedding",
        )
        scored = [
            (item.chunk, _cosine_similarity(query_vector, item.vector))
            for item in visible
        ]
        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return tuple(
            RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank,
                method=self.method,
                score_kind=ScoreKind.COSINE_SIMILARITY,
            )
            for rank, (chunk, score) in enumerate(
                scored[: request.top_k],
                start=1,
            )
        )
