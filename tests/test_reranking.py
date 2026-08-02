from __future__ import annotations

import math
import sys
import types
from collections.abc import Sequence

import pytest

from atlasrag.embeddings.base import EmbeddingModel, Vector
from atlasrag.ingestion.chunking import FixedCharacterChunker
from atlasrag.models import Chunk, Document
from atlasrag.retrieval import (
    AccessPrincipal,
    BM25Retriever,
    Citation,
    CrossEncoderReranker,
    ExactDenseRetriever,
    PermissionPolicy,
    ReciprocalRankFusionRetriever,
    RerankedRetriever,
    Reranker,
    RetrievalMethod,
    RetrievalQuery,
    ScoreKind,
)


class AliasEmbedding(EmbeddingModel):
    @property
    def model_id(self) -> str:
        return "alias-test-v1"

    @property
    def dimension(self) -> int:
        return 3

    def _embed(self, text: str) -> Vector:
        lowered = text.casefold()
        return (
            float("mars" in lowered or "ares" in lowered),
            float("ocean" in lowered or "sea" in lowered),
            1.0,
        )

    def embed_documents(self, texts):
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, text):
        return self._embed(text)


class MappingReranker(Reranker):
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    @property
    def model_id(self) -> str:
        return "mapping-reranker-v1"

    def score(self, query: str, chunks: Sequence[Chunk]) -> tuple[float, ...]:
        return tuple(self._scores[chunk.source_uri] for chunk in chunks)


def _chunk(
    source: str,
    text: str,
    *,
    policy: PermissionPolicy | None = None,
) -> Chunk:
    document = Document.from_text(
        source_uri=f"memory://{source}",
        text=text,
        metadata={} if policy is None else policy.to_metadata(),
    )
    return FixedCharacterChunker(chunk_size=1000).chunk(document)[0]


def _hybrid(chunks: Sequence[Chunk]) -> ReciprocalRankFusionRetriever:
    retriever = ReciprocalRankFusionRetriever(
        BM25Retriever(),
        ExactDenseRetriever(AliasEmbedding()),
        candidate_k=10,
    )
    retriever.index(chunks)
    return retriever


def test_reranker_reorders_candidates_and_preserves_candidate_evidence() -> None:
    lexical = _chunk("lexical", "mars rover mission")
    semantic = _chunk("semantic", "ares exploration program")
    base = _hybrid((lexical, semantic))
    reranked = RerankedRetriever(
        base,
        MappingReranker(
            {
                "memory://lexical": 0.2,
                "memory://semantic": 0.9,
            }
        ),
        candidate_k=2,
    )

    results = reranked.search("mars", top_k=2)

    assert [result.chunk.source_uri for result in results] == [
        "memory://semantic",
        "memory://lexical",
    ]
    assert results[0].method is RetrievalMethod.RERANKED
    assert results[0].score_kind is ScoreKind.RERANKER_RELEVANCE
    assert results[0].rerank_trace is not None
    assert results[0].rerank_trace.model_id == "mapping-reranker-v1"
    assert results[0].rerank_trace.candidate_method is RetrievalMethod.HYBRID_RRF
    assert results[0].rerank_trace.candidate_rank == 2
    assert (
        results[0].rerank_trace.candidate_score_kind is ScoreKind.RECIPROCAL_RANK_FUSION
    )
    assert results[0].contributions == results[0].rerank_trace.candidate_contributions


def test_reranking_preserves_exact_citation_and_metadata() -> None:
    policy = PermissionPolicy(
        tenant_id="tenant-a",
        allowed_groups=frozenset({"ops"}),
    )
    chunk = _chunk("private", "mars private evidence", policy=policy)
    base = _hybrid((chunk,))
    result = RerankedRetriever(
        base,
        MappingReranker({"memory://private": 1.0}),
        candidate_k=1,
    ).search(
        "mars",
        top_k=1,
        principal=AccessPrincipal(
            tenant_id="tenant-a",
            groups=frozenset({"ops"}),
        ),
    )[0]

    assert result.citation == Citation.from_chunk(chunk)
    assert result.citation.source_uri == chunk.source_uri
    assert result.citation.start_char == chunk.start_char
    assert result.citation.end_char == chunk.end_char
    assert result.citation.document_content_sha256 == chunk.document_content_sha256
    assert result.citation.metadata == policy.to_metadata()
    with pytest.raises(TypeError):
        result.citation.metadata["new"] = "value"  # type: ignore[index]


def test_reranker_cannot_reintroduce_unauthorized_candidates() -> None:
    public = _chunk("public", "mars public")
    private = _chunk(
        "private",
        "mars private secret",
        policy=PermissionPolicy(tenant_id="tenant-a"),
    )
    base = _hybrid((public, private))
    reranked = RerankedRetriever(
        base,
        MappingReranker(
            {
                "memory://public": 0.0,
                "memory://private": 100.0,
            }
        ),
        candidate_k=10,
    )

    results = reranked.search("mars", top_k=10)

    assert [result.chunk.source_uri for result in results] == ["memory://public"]


def test_reranking_propagates_exclusions_before_candidate_generation() -> None:
    first = _chunk("first", "mars first")
    second = _chunk("second", "mars second")
    reranked = RerankedRetriever(
        _hybrid((first, second)),
        MappingReranker({"memory://first": 1.0, "memory://second": 0.5}),
        candidate_k=2,
    )

    results = reranked.search(
        RetrievalQuery(
            text="mars",
            top_k=2,
            excluded_chunk_ids=frozenset({first.chunk_id}),
        )
    )

    assert [result.chunk.chunk_id for result in results] == [second.chunk_id]


def test_equal_reranker_scores_preserve_candidate_rank_order() -> None:
    chunks = (_chunk("one", "mars one"), _chunk("two", "mars two"))
    base = _hybrid(chunks)
    candidates = base.search("mars", top_k=2)
    reranked = RerankedRetriever(
        base,
        MappingReranker({chunk.source_uri: 1.0 for chunk in chunks}),
        candidate_k=2,
    )

    results = reranked.search("mars", top_k=2)

    assert [result.chunk.chunk_id for result in results] == [
        candidate.chunk.chunk_id for candidate in candidates
    ]


@pytest.mark.parametrize("candidate_k", [0, -1])
def test_reranked_retriever_rejects_invalid_candidate_depth(candidate_k: int) -> None:
    with pytest.raises(ValueError, match="candidate_k"):
        RerankedRetriever(
            BM25Retriever(),
            MappingReranker({}),
            candidate_k=candidate_k,
        )


class BrokenReranker(Reranker):
    def __init__(self, scores: tuple[float, ...]) -> None:
        self._scores = scores

    @property
    def model_id(self) -> str:
        return "broken"

    def score(self, query: str, chunks: Sequence[Chunk]) -> tuple[float, ...]:
        return self._scores


def test_reranked_retriever_validates_score_count_and_finiteness() -> None:
    base = _hybrid((_chunk("one", "mars one"),))
    with pytest.raises(ValueError, match="one score per candidate"):
        RerankedRetriever(base, BrokenReranker(())).search("mars")
    with pytest.raises(ValueError, match="finite"):
        RerankedRetriever(base, BrokenReranker((math.nan,))).search("mars")


def test_cross_encoder_adapter_builds_query_chunk_pairs(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeCrossEncoder:
        def __init__(
            self,
            model_name: str,
            *,
            revision: str | None,
            device: str | None,
            local_files_only: bool,
        ) -> None:
            calls["model_name"] = model_name
            calls["revision"] = revision
            calls["device"] = device
            calls["local_files_only"] = local_files_only

        def predict(self, pairs, *, batch_size: int):
            calls["pairs"] = pairs
            calls["batch_size"] = batch_size
            return [0.25, 0.75]

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    chunks = (_chunk("one", "first"), _chunk("two", "second"))

    reranker = CrossEncoderReranker(
        "model/test",
        revision="abc123",
        batch_size=8,
        device="cpu",
        local_files_only=True,
    )
    scores = reranker.score("query", chunks)

    assert scores == (0.25, 0.75)
    assert reranker.model_id == "model/test@abc123"
    assert calls == {
        "model_name": "model/test",
        "revision": "abc123",
        "device": "cpu",
        "local_files_only": True,
        "pairs": [("query", "first"), ("query", "second")],
        "batch_size": 8,
    }
