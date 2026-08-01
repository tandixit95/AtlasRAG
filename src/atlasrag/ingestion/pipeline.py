"""Composable ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from atlasrag.ingestion.base import DocumentSource
from atlasrag.ingestion.chunking import ChunkingStrategy
from atlasrag.models import Chunk, Document


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Documents loaded from a source and the chunks derived from them."""

    documents: tuple[Document, ...]
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True, slots=True)
class IngestionPipeline:
    """Compose source loading with an explicit chunking strategy."""

    chunker: ChunkingStrategy

    def run(self, source: DocumentSource) -> IngestionResult:
        documents = tuple(source.load())
        chunks = tuple(
            chunk for document in documents for chunk in self.chunker.chunk(document)
        )
        return IngestionResult(documents=documents, chunks=chunks)
