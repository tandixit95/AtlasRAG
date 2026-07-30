"""Source adapters and ingestion transformations for AtlasRAG."""

from atlasrag.ingestion.base import DocumentSource
from atlasrag.ingestion.chunking import ChunkingStrategy, FixedCharacterChunker
from atlasrag.ingestion.pipeline import IngestionPipeline, IngestionResult
from atlasrag.ingestion.text import PlainTextSource

__all__ = [
    "ChunkingStrategy",
    "DocumentSource",
    "FixedCharacterChunker",
    "IngestionPipeline",
    "IngestionResult",
    "PlainTextSource",
]
