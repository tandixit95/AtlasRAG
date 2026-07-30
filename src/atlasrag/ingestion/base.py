"""Ingestion contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from atlasrag.models import Document


class DocumentSource(ABC):
    """Boundary for loading source material into AtlasRAG documents."""

    @abstractmethod
    def load(self) -> Sequence[Document]:
        """Load zero or more documents from this source."""
