"""Source adapters for AtlasRAG ingestion."""

from atlasrag.ingestion.base import DocumentSource
from atlasrag.ingestion.text import PlainTextSource

__all__ = ["DocumentSource", "PlainTextSource"]
