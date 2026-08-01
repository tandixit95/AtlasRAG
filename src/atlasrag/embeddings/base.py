"""Embedding contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

Vector = tuple[float, ...]


class EmbeddingModel(ABC):
    """Model-independent boundary for document and query embeddings."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]: ...

    @abstractmethod
    def embed_query(self, text: str) -> Vector: ...
