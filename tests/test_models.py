from __future__ import annotations

import pytest

from atlasrag.models import Document


def test_document_identity_is_stable_for_same_source_uri() -> None:
    first = Document.from_text(source_uri="file:///docs/manual.txt", text="version one")
    second = Document.from_text(source_uri="file:///docs/manual.txt", text="version two")

    assert first.document_id == second.document_id


def test_content_hash_changes_when_text_changes() -> None:
    first = Document.from_text(source_uri="file:///docs/manual.txt", text="version one")
    second = Document.from_text(source_uri="file:///docs/manual.txt", text="version two")

    assert first.content_sha256 != second.content_sha256


def test_document_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="text must not be blank"):
        Document.from_text(source_uri="file:///docs/manual.txt", text="   \n")


def test_metadata_is_snapshotted_and_read_only() -> None:
    metadata = {"owner": "docs-team"}
    document = Document.from_text(
        source_uri="file:///docs/manual.txt",
        text="payload",
        metadata=metadata,
    )
    metadata["owner"] = "changed"

    assert document.metadata["owner"] == "docs-team"
    with pytest.raises(TypeError):
        document.metadata["owner"] = "mutated"  # type: ignore[index]


def test_document_rejects_content_hash_that_does_not_match_text() -> None:
    with pytest.raises(ValueError, match="must match document text"):
        Document(
            document_id="doc-1",
            source_uri="memory://doc-1",
            text="payload",
            content_sha256="0" * 64,
        )
