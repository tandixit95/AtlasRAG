"""Core domain models for AtlasRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping
from uuid import NAMESPACE_URL, uuid5


def _sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a SHA-256 hex digest") from exc


@dataclass(frozen=True, slots=True)
class Document:
    """A logical source document and the exact text version ingested from it.

    ``document_id`` identifies the logical source. ``content_sha256`` identifies
    the exact text payload. Keeping these separate allows ingestion code to
    distinguish a source update from a newly discovered source.
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
        _validate_sha256(self.content_sha256, field_name="content_sha256")
        if self.content_sha256 != _sha256_text(self.text):
            raise ValueError("content_sha256 must match document text")
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
        return cls(
            document_id=resolved_id,
            source_uri=normalized_uri,
            text=text,
            content_sha256=_sha256_text(text),
            metadata=metadata or {},
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    """An exact character span produced from a particular document version.

    Character offsets are half-open: ``start_char`` is inclusive and
    ``end_char`` is exclusive. The chunk ID includes source identity, chunking
    strategy, span, and chunk contents, but not the whole-document content
    digest. An unchanged span can therefore retain its ID when unrelated text
    elsewhere in the document changes, while ``document_content_sha256`` still
    records the exact source version that produced it.
    """

    chunk_id: str
    document_id: str
    document_content_sha256: str
    source_uri: str
    text: str
    content_sha256: str
    start_char: int
    end_char: int
    ordinal: int
    strategy_id: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must not be blank")
        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")
        if not self.source_uri.strip():
            raise ValueError("source_uri must not be blank")
        if self.text == "":
            raise ValueError("text must not be empty")
        _validate_sha256(
            self.document_content_sha256,
            field_name="document_content_sha256",
        )
        _validate_sha256(self.content_sha256, field_name="content_sha256")
        if self.content_sha256 != _sha256_text(self.text):
            raise ValueError("content_sha256 must match chunk text")
        if self.start_char < 0:
            raise ValueError("start_char must be non-negative")
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("character span length must match chunk text length")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id must not be blank")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_document_span(
        cls,
        *,
        document: Document,
        start_char: int,
        end_char: int,
        ordinal: int,
        strategy_id: str,
    ) -> Chunk:
        """Create a chunk from an exact span in ``document``."""

        if start_char < 0 or end_char > len(document.text):
            raise ValueError("chunk span must be within document text")
        if end_char <= start_char:
            raise ValueError("end_char must be greater than start_char")

        text = document.text[start_char:end_char]
        chunk_digest = _sha256_text(text)
        identity = ":".join(
            (
                "atlasrag-chunk-v1",
                document.document_id,
                strategy_id,
                str(start_char),
                str(end_char),
                chunk_digest,
            )
        )
        return cls(
            chunk_id=str(uuid5(NAMESPACE_URL, identity)),
            document_id=document.document_id,
            document_content_sha256=document.content_sha256,
            source_uri=document.source_uri,
            text=text,
            content_sha256=chunk_digest,
            start_char=start_char,
            end_char=end_char,
            ordinal=ordinal,
            strategy_id=strategy_id,
            metadata=document.metadata,
        )
