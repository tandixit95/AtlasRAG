"""Permission metadata and fail-closed visibility checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from atlasrag.models import Chunk
from atlasrag.retrieval.contracts import AccessPrincipal

TENANT_METADATA_KEY = "atlasrag.access.tenant"
GROUPS_METADATA_KEY = "atlasrag.access.groups"


def _normalize_value(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _normalize_groups(groups: frozenset[str]) -> frozenset[str]:
    if isinstance(groups, str):
        raise TypeError(
            "allowed_groups must be an iterable of group names, not a string"
        )
    normalized: set[str] = set()
    for group in groups:
        value = _normalize_value(group, field_name="allowed group")
        if "," in value:
            raise ValueError("allowed group names must not contain commas")
        normalized.add(value)
    return frozenset(normalized)


@dataclass(frozen=True, slots=True)
class PermissionPolicy:
    """Conjunctive tenant and group requirements attached to a chunk.

    A policy with no tenant and no groups is public. When a tenant is present,
    the caller must match it. When groups are present, the caller must belong to
    at least one. If both are present, both checks must pass.
    """

    tenant_id: str | None = None
    allowed_groups: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        tenant_id = self.tenant_id
        if tenant_id is not None:
            tenant_id = _normalize_value(tenant_id, field_name="tenant_id")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(
            self,
            "allowed_groups",
            _normalize_groups(self.allowed_groups),
        )

    @property
    def is_public(self) -> bool:
        return self.tenant_id is None and not self.allowed_groups

    def allows(self, principal: AccessPrincipal) -> bool:
        if self.tenant_id is not None and principal.tenant_id != self.tenant_id:
            return False
        if self.allowed_groups and self.allowed_groups.isdisjoint(principal.groups):
            return False
        return True

    def to_metadata(self) -> dict[str, str]:
        """Return deterministic metadata suitable for ``Document.from_text``."""

        metadata: dict[str, str] = {}
        if self.tenant_id is not None:
            metadata[TENANT_METADATA_KEY] = self.tenant_id
        if self.allowed_groups:
            metadata[GROUPS_METADATA_KEY] = ",".join(sorted(self.allowed_groups))
        return metadata

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, str]) -> PermissionPolicy:
        """Parse access metadata, rejecting malformed protected policies."""

        tenant_id = metadata.get(TENANT_METADATA_KEY)
        raw_groups = metadata.get(GROUPS_METADATA_KEY)
        groups: frozenset[str] = frozenset()
        if raw_groups is not None:
            if not raw_groups.strip():
                raise ValueError(f"{GROUPS_METADATA_KEY} must not be blank")
            parts = raw_groups.split(",")
            if any(not part.strip() for part in parts):
                raise ValueError(
                    f"{GROUPS_METADATA_KEY} must be a comma-separated group list"
                )
            groups = frozenset(part.strip() for part in parts)
        return cls(tenant_id=tenant_id, allowed_groups=groups)


def policy_for_chunk(chunk: Chunk) -> PermissionPolicy:
    """Return the validated policy associated with ``chunk``."""

    return PermissionPolicy.from_metadata(chunk.metadata)
