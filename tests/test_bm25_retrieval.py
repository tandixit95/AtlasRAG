from __future__ import annotations

import pytest

from atlasrag.ingestion.chunking import FixedCharacterChunker
from atlasrag.models import Chunk, Document
from atlasrag.retrieval import (
    AccessPrincipal,
    BM25Retriever,
    PermissionPolicy,
    RetrievalMethod,
    RetrievalQuery,
    ScoreKind,
)


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


def test_bm25_ranks_exact_term_frequency_and_preserves_provenance() -> None:
    chunks = (
        _chunk("mars", "mars rover mars mission"),
        _chunk("ocean", "ocean current research"),
        _chunk("mixed", "mars ocean survey"),
    )
    retriever = BM25Retriever()
    retriever.index(chunks)

    results = retriever.search("mars", top_k=2)

    assert [result.chunk.source_uri for result in results] == [
        "memory://mars",
        "memory://mixed",
    ]
    assert results[0].chunk == chunks[0]
    assert results[0].method is RetrievalMethod.BM25
    assert results[0].score_kind is ScoreKind.BM25
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score > results[1].score


def test_bm25_returns_only_matching_chunks() -> None:
    retriever = BM25Retriever()
    retriever.index((_chunk("one", "forest canopy"), _chunk("two", "ocean tide")))

    assert retriever.search("mars") == ()
    assert retriever.search("!!!") == ()


def test_bm25_ties_are_stable_by_chunk_id() -> None:
    chunks = (_chunk("one", "same token"), _chunk("two", "same token"))
    retriever = BM25Retriever()
    retriever.index(chunks)

    first = retriever.search("same")
    second = retriever.search("same")

    expected = sorted(chunk.chunk_id for chunk in chunks)
    assert [result.chunk.chunk_id for result in first] == expected
    assert [result.chunk.chunk_id for result in second] == expected


def test_bm25_enforces_tenant_and_group_permissions() -> None:
    chunks = (
        _chunk("public", "mars public"),
        _chunk(
            "tenant-a",
            "mars tenant secret",
            policy=PermissionPolicy(tenant_id="tenant-a"),
        ),
        _chunk(
            "tenant-a-ops",
            "mars operations secret",
            policy=PermissionPolicy(
                tenant_id="tenant-a",
                allowed_groups=frozenset({"ops"}),
            ),
        ),
        _chunk(
            "tenant-b",
            "mars other tenant secret",
            policy=PermissionPolicy(tenant_id="tenant-b"),
        ),
    )
    retriever = BM25Retriever()
    retriever.index(chunks)

    anonymous = retriever.search("mars", top_k=10)
    tenant_reader = retriever.search(
        RetrievalQuery(
            "mars",
            top_k=10,
            principal=AccessPrincipal(tenant_id="tenant-a"),
        )
    )
    tenant_ops = retriever.search(
        "mars",
        top_k=10,
        principal=AccessPrincipal(
            tenant_id="tenant-a",
            groups=frozenset({"ops"}),
        ),
    )

    assert {result.chunk.source_uri for result in anonymous} == {"memory://public"}
    assert {result.chunk.source_uri for result in tenant_reader} == {
        "memory://public",
        "memory://tenant-a",
    }
    assert {result.chunk.source_uri for result in tenant_ops} == {
        "memory://public",
        "memory://tenant-a",
        "memory://tenant-a-ops",
    }
    assert all(result.chunk.source_uri != "memory://tenant-b" for result in tenant_ops)


def test_unauthorized_chunks_do_not_change_visible_bm25_scores() -> None:
    visible = _chunk("public", "mars rover")
    unauthorized = _chunk(
        "private",
        "mars mars mars mars mars",
        policy=PermissionPolicy(tenant_id="tenant-b"),
    )
    isolated = BM25Retriever()
    isolated.index((visible,))
    mixed = BM25Retriever()
    mixed.index((visible, unauthorized))

    isolated_result = isolated.search("mars")[0]
    mixed_result = mixed.search("mars")[0]

    assert mixed_result.chunk == isolated_result.chunk
    assert mixed_result.score == pytest.approx(isolated_result.score)


def test_bm25_rejects_invalid_configuration_and_duplicate_chunks() -> None:
    with pytest.raises(ValueError, match="k1"):
        BM25Retriever(k1=0)
    with pytest.raises(ValueError, match="between"):
        BM25Retriever(b=1.1)

    chunk = _chunk("one", "mars")
    with pytest.raises(ValueError, match="unique"):
        BM25Retriever().index((chunk, chunk))


def test_bm25_rejects_overrides_with_typed_query() -> None:
    retriever = BM25Retriever()
    with pytest.raises(TypeError, match="overrides"):
        retriever.search(RetrievalQuery("mars"), top_k=2)


def test_bm25_search_before_index_returns_empty_results() -> None:
    assert BM25Retriever().search("mars") == ()
