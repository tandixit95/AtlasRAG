"""Chunking contracts and baseline strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from atlasrag.models import Chunk, Document


class ChunkingStrategy(ABC):
    """Boundary for deterministic document-to-chunk transformations."""

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Stable identifier for the strategy implementation and configuration."""

    @abstractmethod
    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Transform one document into ordered chunks."""


@dataclass(frozen=True, slots=True)
class FixedCharacterChunker(ChunkingStrategy):
    """Split documents into fixed-size character windows with optional overlap.

    This is intentionally a transparent baseline, not a claim that character
    chunking is optimal for retrieval. It provides deterministic boundaries for
    testing the ingestion contract before tokenizer- or structure-aware
    strategies are introduced and measured.
    """

    chunk_size: int
    overlap: int = 0

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

    @property
    def strategy_id(self) -> str:
        return f"fixed-character-v1:size={self.chunk_size}:overlap={self.overlap}"

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        start = 0

        while start < len(document.text):
            end = min(start + self.chunk_size, len(document.text))
            chunks.append(
                Chunk.from_document_span(
                    document=document,
                    start_char=start,
                    end_char=end,
                    ordinal=len(chunks),
                    strategy_id=self.strategy_id,
                )
            )
            if end == len(document.text):
                break
            start = end - self.overlap

        return tuple(chunks)
