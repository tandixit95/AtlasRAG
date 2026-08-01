"""Dependency-light, permission-aware BM25 retrieval baseline."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import log

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

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Return a deterministic, Unicode-aware lexical baseline tokenization."""

    return tuple(match.group(0) for match in _TOKEN_PATTERN.finditer(text.casefold()))


@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    chunk: Chunk
    term_frequencies: Counter[str]
    token_count: int
    policy: PermissionPolicy


class BM25Retriever(Retriever):
    """Local BM25 baseline with authorization-scoped corpus statistics.

    Corpus statistics are recomputed over visible chunks for each principal.
    This is intentionally a small-corpus correctness and security baseline: an
    unauthorized chunk cannot be returned and cannot perturb an authorized
    caller's document frequencies, length normalization, score, or rank.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0.0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self._k1 = float(k1)
        self._b = float(b)
        self._indexed: tuple[_IndexedChunk, ...] = ()

    @property
    def method(self) -> RetrievalMethod:
        return RetrievalMethod.BM25

    @property
    def k1(self) -> float:
        return self._k1

    @property
    def b(self) -> float:
        return self._b

    def index(self, chunks: Sequence[Chunk]) -> None:
        chunk_tuple = tuple(chunks)
        chunk_ids = [chunk.chunk_id for chunk in chunk_tuple]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk IDs must be unique within an index")
        self._indexed = tuple(
            _IndexedChunk(
                chunk=chunk,
                term_frequencies=Counter(tokens := tokenize(chunk.text)),
                token_count=len(tokens),
                policy=policy_for_chunk(chunk),
            )
            for chunk in chunk_tuple
        )

    def search(
        self,
        query: str | RetrievalQuery,
        *,
        top_k: int | None = None,
        principal: AccessPrincipal | None = None,
    ) -> tuple[RetrievalResult, ...]:
        request = resolve_query(query, top_k=top_k, principal=principal)
        query_terms = Counter(tokenize(request.text))
        if not query_terms:
            return ()

        visible = tuple(
            item for item in self._indexed if item.policy.allows(request.principal)
        )
        if not visible:
            return ()

        document_count = len(visible)
        average_length = sum(item.token_count for item in visible) / document_count
        document_frequency = {
            term: sum(term in item.term_frequencies for item in visible)
            for term in query_terms
        }

        scored: list[tuple[Chunk, float]] = []
        for item in visible:
            score = 0.0
            length_ratio = (
                item.token_count / average_length if average_length > 0.0 else 0.0
            )
            for term, query_frequency in query_terms.items():
                term_frequency = item.term_frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                frequency = document_frequency[term]
                inverse_document_frequency = log(
                    1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
                )
                denominator = term_frequency + self._k1 * (
                    1.0 - self._b + self._b * length_ratio
                )
                score += (
                    query_frequency
                    * inverse_document_frequency
                    * (term_frequency * (self._k1 + 1.0) / denominator)
                )
            if score > 0.0:
                scored.append((item.chunk, score))

        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return tuple(
            RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank,
                method=self.method,
                score_kind=ScoreKind.BM25,
            )
            for rank, (chunk, score) in enumerate(
                scored[: request.top_k],
                start=1,
            )
        )
