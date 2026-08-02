from __future__ import annotations

import pytest

from atlasrag.retrieval import (
    AccessPrincipal,
    PermissionPolicy,
    RetrievalQuery,
)


def test_access_principal_normalizes_groups_and_tenant() -> None:
    principal = AccessPrincipal(tenant_id=" tenant-a ", groups={" ops ", "readers"})

    assert principal.tenant_id == "tenant-a"
    assert principal.groups == frozenset({"ops", "readers"})


def test_permission_policy_round_trips_deterministic_metadata() -> None:
    policy = PermissionPolicy(
        tenant_id="tenant-a",
        allowed_groups=frozenset({"readers", "ops"}),
    )

    assert policy.to_metadata() == {
        "atlasrag.access.tenant": "tenant-a",
        "atlasrag.access.groups": "ops,readers",
    }
    assert PermissionPolicy.from_metadata(policy.to_metadata()) == policy


def test_permission_policy_requires_all_present_dimensions() -> None:
    policy = PermissionPolicy(
        tenant_id="tenant-a",
        allowed_groups=frozenset({"readers"}),
    )

    assert policy.allows(
        AccessPrincipal(tenant_id="tenant-a", groups=frozenset({"readers"}))
    )
    assert not policy.allows(
        AccessPrincipal(tenant_id="tenant-b", groups=frozenset({"readers"}))
    )
    assert not policy.allows(
        AccessPrincipal(tenant_id="tenant-a", groups=frozenset({"writers"}))
    )


def test_public_policy_allows_anonymous_principal() -> None:
    assert PermissionPolicy().allows(AccessPrincipal())


@pytest.mark.parametrize(
    "metadata",
    [
        {"atlasrag.access.tenant": "   "},
        {"atlasrag.access.groups": ""},
        {"atlasrag.access.groups": "ops,,readers"},
    ],
)
def test_malformed_permission_metadata_fails_closed(metadata: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        PermissionPolicy.from_metadata(metadata)


def test_retrieval_query_validates_text_and_top_k() -> None:
    with pytest.raises(ValueError, match="blank"):
        RetrievalQuery("  ")
    with pytest.raises(ValueError, match="top_k"):
        RetrievalQuery("valid", top_k=0)


def test_group_collections_reject_bare_strings() -> None:
    with pytest.raises(TypeError, match="not a string"):
        AccessPrincipal(groups="ops")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="not a string"):
        PermissionPolicy(allowed_groups="ops")  # type: ignore[arg-type]


def test_retrieval_query_requires_access_principal() -> None:
    with pytest.raises(TypeError, match="AccessPrincipal"):
        RetrievalQuery("mars", principal=None)  # type: ignore[arg-type]


def test_retrieval_query_normalizes_excluded_chunk_ids() -> None:
    query = RetrievalQuery(text="query", excluded_chunk_ids={" chunk-b ", "chunk-a"})

    assert query.excluded_chunk_ids == frozenset({"chunk-a", "chunk-b"})


@pytest.mark.parametrize("excluded", ["chunk-a", {" "}])
def test_retrieval_query_rejects_invalid_exclusions(excluded) -> None:
    expected = TypeError if isinstance(excluded, str) else ValueError
    with pytest.raises(expected):
        RetrievalQuery(text="query", excluded_chunk_ids=excluded)


def test_citation_rejects_invalid_hashes() -> None:
    from atlasrag.retrieval import Citation

    with pytest.raises(ValueError, match="SHA-256"):
        Citation(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_content_sha256="not-a-digest",
            source_uri="memory://doc-1",
            start_char=0,
            end_char=4,
            content_sha256="0" * 64,
            strategy_id="test-v1",
        )
