from __future__ import annotations

from dataclasses import dataclass

from atlasrag.ingestion.base import DocumentSource
from atlasrag.ingestion.chunking import FixedCharacterChunker
from atlasrag.ingestion.pipeline import IngestionPipeline
from atlasrag.models import Document


@dataclass(frozen=True)
class StubSource(DocumentSource):
    documents: tuple[Document, ...]

    def load(self) -> tuple[Document, ...]:
        return self.documents


def test_pipeline_preserves_documents_and_flattens_chunks_in_source_order() -> None:
    first = Document.from_text(source_uri="memory://first", text="abcdef")
    second = Document.from_text(source_uri="memory://second", text="uvwxyz")
    pipeline = IngestionPipeline(FixedCharacterChunker(chunk_size=4))

    result = pipeline.run(StubSource((first, second)))

    assert result.documents == (first, second)
    assert [chunk.text for chunk in result.chunks] == ["abcd", "ef", "uvwx", "yz"]
    assert [chunk.document_id for chunk in result.chunks] == [
        first.document_id,
        first.document_id,
        second.document_id,
        second.document_id,
    ]


def test_pipeline_handles_source_with_no_documents() -> None:
    result = IngestionPipeline(FixedCharacterChunker(chunk_size=4)).run(StubSource(()))

    assert result.documents == ()
    assert result.chunks == ()


def test_plain_text_source_runs_through_pipeline_with_provenance(tmp_path) -> None:
    from atlasrag.ingestion.text import PlainTextSource

    path = tmp_path / "manual.txt"
    path.write_text("abcdefgh", encoding="utf-8")

    result = IngestionPipeline(FixedCharacterChunker(chunk_size=5, overlap=2)).run(
        PlainTextSource(path)
    )

    assert len(result.documents) == 1
    assert [(chunk.text, chunk.start_char, chunk.end_char) for chunk in result.chunks] == [
        ("abcde", 0, 5),
        ("defgh", 3, 8),
    ]
    assert all(chunk.source_uri == path.resolve().as_uri() for chunk in result.chunks)
    assert all(chunk.metadata["filename"] == "manual.txt" for chunk in result.chunks)
