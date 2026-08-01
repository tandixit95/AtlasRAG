from __future__ import annotations

from atlasrag.embeddings.base import EmbeddingModel, Vector
from atlasrag.ingestion.chunking import FixedCharacterChunker
from atlasrag.models import Chunk, Document
from atlasrag.retrieval import (
    AccessPrincipal,
    BM25Retriever,
    ExactDenseRetriever,
    PermissionPolicy,
    ReciprocalRankFusionRetriever,
    RetrievalMethod,
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
        red_planet = float("mars" in lowered or "ares" in lowered)
        ocean = float("ocean" in lowered or "sea" in lowered)
        return (red_planet, ocean, 1.0)

    def embed_documents(self, texts):
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, text):
        return self._embed(text)


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


def _hybrid() -> ReciprocalRankFusionRetriever:
    return ReciprocalRankFusionRetriever(
        BM25Retriever(),
        ExactDenseRetriever(AliasEmbedding()),
        rrf_k=60,
        candidate_k=10,
    )


def test_hybrid_rrf_combines_ranks_and_preserves_raw_component_metadata() -> None:
    chunks = (
        _chunk("lexical", "mars rover mission"),
        _chunk("semantic", "ares exploration program"),
        _chunk("other", "ocean current research"),
    )
    retriever = _hybrid()
    retriever.index(chunks)

    results = retriever.search("mars", top_k=3)

    assert results[0].method is RetrievalMethod.HYBRID_RRF
    assert results[0].score_kind is ScoreKind.RECIPROCAL_RANK_FUSION
    assert results[0].chunk.source_uri == "memory://lexical"
    assert {item.method for item in results[0].contributions} == {
        RetrievalMethod.BM25,
        RetrievalMethod.EXACT_DENSE,
    }
    assert {item.score_kind for item in results[0].contributions} == {
        ScoreKind.BM25,
        ScoreKind.COSINE_SIMILARITY,
    }
    assert results[0].score == sum(
        1.0 / (60 + item.rank) for item in results[0].contributions
    )


def test_hybrid_can_recover_dense_only_alias_candidate() -> None:
    chunks = (
        _chunk("semantic", "ares exploration program"),
        _chunk("other", "ocean current research"),
    )
    retriever = _hybrid()
    retriever.index(chunks)

    results = retriever.search("mars", top_k=2)

    semantic = next(
        result for result in results if result.chunk.source_uri == "memory://semantic"
    )
    assert {item.method for item in semantic.contributions} == {
        RetrievalMethod.EXACT_DENSE
    }


def test_hybrid_permission_filter_blocks_leakage_from_every_component() -> None:
    chunks = (
        _chunk("public", "mars public"),
        _chunk(
            "private",
            "mars ares private secret",
            policy=PermissionPolicy(
                tenant_id="tenant-a",
                allowed_groups=frozenset({"ops"}),
            ),
        ),
    )
    retriever = _hybrid()
    retriever.index(chunks)

    anonymous = retriever.search("mars", top_k=10)
    wrong_group = retriever.search(
        "mars",
        top_k=10,
        principal=AccessPrincipal(
            tenant_id="tenant-a",
            groups=frozenset({"readers"}),
        ),
    )
    authorized = retriever.search(
        "mars",
        top_k=10,
        principal=AccessPrincipal(
            tenant_id="tenant-a",
            groups=frozenset({"ops"}),
        ),
    )

    assert {result.chunk.source_uri for result in anonymous} == {"memory://public"}
    assert {result.chunk.source_uri for result in wrong_group} == {"memory://public"}
    assert {result.chunk.source_uri for result in authorized} == {
        "memory://public",
        "memory://private",
    }


def test_hybrid_ties_are_deterministic() -> None:
    chunks = (_chunk("one", "same"), _chunk("two", "same"))
    retriever = _hybrid()
    retriever.index(chunks)

    first = retriever.search("same", top_k=2)
    second = retriever.search("same", top_k=2)

    assert [result.chunk.chunk_id for result in first] == [
        result.chunk.chunk_id for result in second
    ]


def test_hybrid_search_before_index_returns_empty_results() -> None:
    assert _hybrid().search("mars") == ()


def test_hybrid_rejects_invalid_fusion_configuration() -> None:
    import pytest

    with pytest.raises(ValueError, match="rrf_k"):
        ReciprocalRankFusionRetriever(
            BM25Retriever(),
            ExactDenseRetriever(AliasEmbedding()),
            rrf_k=0,
        )
    with pytest.raises(ValueError, match="candidate_k"):
        ReciprocalRankFusionRetriever(
            BM25Retriever(),
            ExactDenseRetriever(AliasEmbedding()),
            candidate_k=0,
        )
