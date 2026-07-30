"""Core domain models for AtlasRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping
from uuid import NAMESPACE_URL, uuid5


@dataclass(frozen=True, slots=True)
class Document:
    """A logical source document and the exact text version ingested from it.

    ``document_id`` identifies the logical source. ``content_sha256`` identifies
    the exact text payload. Keeping these separate allows future ingestion code
    to distinguish a source update from a newly discovered source.
    """

    document_id: str
    source_uri: str
    text: str
    content_sha256: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")
        if not self.source_uri.strip():
            raise ValueError("source_uri must not be blank")
        if not self.text.strip():
            raise ValueError("text must not be blank")
        if len(self.content_sha256) != 64:
            raise ValueError("content_sha256 must be a SHA-256 hex digest")
        try:
            int(self.content_sha256, 16)
        except ValueError as exc:
            raise ValueError("content_sha256 must be a SHA-256 hex digest") from exc

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_text(
        cls,
        *,
        source_uri: str,
        text: str,
        metadata: Mapping[str, str] | None = None,
        document_id: str | None = None,
    ) -> Document:
        """Create a document with deterministic identity and content hashing."""

        normalized_uri = source_uri.strip()
        if not normalized_uri:
            raise ValueError("source_uri must not be blank")
        if not text.strip():
            raise ValueError("text must not be blank")

        resolved_id = document_id or str(uuid5(NAMESPACE_URL, normalized_uri))
        digest = sha256(text.encode("utf-8")).hexdigest()
        return cls(
            document_id=resolved_id,
            source_uri=normalized_uri,
            text=text,
            content_sha256=digest,
            metadata=metadata or {},
        )
