from __future__ import annotations

from pathlib import Path

import pytest

from atlasrag.ingestion.text import PlainTextSource


def test_plain_text_source_preserves_text_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "guide.txt"
    expected = "  Keep source whitespace.\nSecond line.\n"
    path.write_text(expected, encoding="utf-8")

    documents = PlainTextSource(path).load()

    assert len(documents) == 1
    document = documents[0]
    assert document.text == expected
    assert document.source_uri == path.resolve().as_uri()
    assert document.metadata == {
        "filename": "guide.txt",
        "media_type": "text/plain",
        "encoding": "utf-8",
    }


def test_plain_text_source_raises_for_missing_file(tmp_path: Path) -> None:
    source = PlainTextSource(tmp_path / "missing.txt")

    with pytest.raises(FileNotFoundError):
        source.load()


def test_plain_text_source_rejects_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a file"):
        PlainTextSource(tmp_path).load()
