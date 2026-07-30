from __future__ import annotations

import pytest

from atlasrag.ingestion.chunking import FixedCharacterChunker
from atlasrag.models import Document


def _document(text: str = "abcdefghij") -> Document:
    return Document.from_text(
        source_uri="file:///docs/manual.txt",
        text=text,
        metadata={"owner": "docs-team"},
    )


def test_fixed_character_chunker_emits_exact_non_overlapping_spans() -> None:
    chunks = FixedCharacterChunker(chunk_size=4).chunk(_document())

    assert [(chunk.text, chunk.start_char, chunk.end_char) for chunk in chunks] == [
        ("abcd", 0, 4),
        ("efgh", 4, 8),
        ("ij", 8, 10),
    ]
    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2]


def test_fixed_character_chunker_applies_overlap_without_duplicate_tail() -> None:
    chunks = FixedCharacterChunker(chunk_size=6, overlap=2).chunk(_document())

    assert [(chunk.text, chunk.start_char, chunk.end_char) for chunk in chunks] == [
        ("abcdef", 0, 6),
        ("efghij", 4, 10),
    ]


def test_chunk_propagates_source_version_and_metadata() -> None:
    document = _document()
    chunk = FixedCharacterChunker(chunk_size=4).chunk(document)[0]

    assert chunk.document_id == document.document_id
    assert chunk.document_content_sha256 == document.content_sha256
    assert chunk.source_uri == document.source_uri
    assert chunk.metadata == document.metadata
    assert chunk.strategy_id == "fixed-character-v1:size=4:overlap=0"


def test_chunk_ids_are_deterministic_for_same_document_and_config() -> None:
    document = _document()
    chunker = FixedCharacterChunker(chunk_size=4, overlap=1)

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunk_id_changes_when_chunking_configuration_changes() -> None:
    document = _document()

    first = FixedCharacterChunker(chunk_size=4).chunk(document)[0]
    second = FixedCharacterChunker(chunk_size=5).chunk(document)[0]

    assert first.chunk_id != second.chunk_id


def test_unchanged_span_keeps_chunk_id_when_other_document_content_changes() -> None:
    first_document = _document("abcdefghij")
    second_document = _document("abcdefghXX")
    chunker = FixedCharacterChunker(chunk_size=4)

    first_chunk = chunker.chunk(first_document)[0]
    second_chunk = chunker.chunk(second_document)[0]

    assert first_chunk.chunk_id == second_chunk.chunk_id
    assert first_chunk.document_content_sha256 != second_chunk.document_content_sha256


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (0, 0, "chunk_size must be positive"),
        (-1, 0, "chunk_size must be positive"),
        (4, -1, "overlap must be non-negative"),
        (4, 4, "overlap must be smaller than chunk_size"),
        (4, 5, "overlap must be smaller than chunk_size"),
    ],
)
def test_fixed_character_chunker_rejects_invalid_configuration(
    chunk_size: int,
    overlap: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FixedCharacterChunker(chunk_size=chunk_size, overlap=overlap)


def test_chunk_text_matches_recorded_character_span() -> None:
    document = _document("αβγδεζηθ")
    chunks = FixedCharacterChunker(chunk_size=3, overlap=1).chunk(document)

    for chunk in chunks:
        assert chunk.text == document.text[chunk.start_char : chunk.end_char]
