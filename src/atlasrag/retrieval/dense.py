"""Exact dense retrieval baseline."""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from collections.abc import Sequence
from atlasrag.embeddings.base import EmbeddingModel, Vector
from atlasrag.models import Chunk

@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked retrieval result with its source chunk intact."""
    chunk: Chunk
    score: float
    rank: int

@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    chunk: Chunk
    vector: Vector


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

class ExactDenseRetriever:
    """In-memory exhaustive cosine search used as the correctness baseline."""
    def __init__(self, embedder: EmbeddingModel) -> None:
        self._embedder = embedder
        self._indexed: tuple[_IndexedChunk, ...] = ()

    @property
    def model_id(self) -> str:
        return self._embedder.model_id

    def index(self, chunks: Sequence[Chunk]) -> None:
        vectors = tuple(self._embedder.embed_documents([chunk.text for chunk in chunks]))
        if len(vectors) != len(chunks):
            raise ValueError("embedder must return exactly one vector per chunk")
        for vector in vectors:
            if len(vector) != self._embedder.dimension:
                raise ValueError("embedding dimension does not match embedder contract")
        self._indexed = tuple(_IndexedChunk(chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True))

    def search(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            raise ValueError("query must not be blank")
        if not self._indexed:
            return ()
        query_vector = self._embedder.embed_query(query)
        if len(query_vector) != self._embedder.dimension:
            raise ValueError("query embedding dimension does not match embedder contract")
        scored = [(item.chunk, _cosine_similarity(query_vector, item.vector)) for item in self._indexed]
        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return tuple(
            RetrievalResult(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
        )
