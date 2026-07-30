"""Plain-text ingestion adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlasrag.ingestion.base import DocumentSource
from atlasrag.models import Document


@dataclass(frozen=True, slots=True)
class PlainTextSource(DocumentSource):
    """Load one local text file while preserving its exact decoded contents."""

    path: Path
    encoding: str = "utf-8"

    def load(self) -> tuple[Document, ...]:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"text source must be a file: {path}")

        resolved_path = path.resolve()
        text = resolved_path.read_text(encoding=self.encoding)
        document = Document.from_text(
            source_uri=resolved_path.as_uri(),
            text=text,
            metadata={
                "filename": resolved_path.name,
                "media_type": "text/plain",
                "encoding": self.encoding,
            },
        )
        return (document,)
