"""An offline, model-free demonstration of the actual retrieval access boundary."""

from __future__ import annotations

import json

from atlasrag.ingestion import FixedCharacterChunker
from atlasrag.models import Document
from atlasrag.retrieval import (
    AccessPrincipal,
    BM25Retriever,
    PermissionPolicy,
    RetrievalQuery,
)


def run_demo() -> dict[str, list[str]]:
    documents = [
        Document.from_text(source_uri="memory://public", text="Mars mission guide"),
        Document.from_text(
            source_uri="memory://tenant-a",
            text="Mars mission private operations",
            metadata=PermissionPolicy(
                tenant_id="tenant-a", allowed_groups=frozenset({"ops"})
            ).to_metadata(),
        ),
        Document.from_text(
            source_uri="memory://tenant-b",
            text="Mars mission other tenant operations",
            metadata=PermissionPolicy(tenant_id="tenant-b").to_metadata(),
        ),
    ]
    chunker = FixedCharacterChunker(chunk_size=500)
    retriever = BM25Retriever()
    retriever.index(tuple(chunk for doc in documents for chunk in chunker.chunk(doc)))
    principals = {
        "anonymous": AccessPrincipal(),
        "tenant_a_ops": AccessPrincipal(
            tenant_id="tenant-a", groups=frozenset({"ops"})
        ),
        "tenant_a_wrong_group": AccessPrincipal(
            tenant_id="tenant-a", groups=frozenset({"sales"})
        ),
    }
    return {
        name: sorted(
            result.citation.source_uri
            for result in retriever.search(
                RetrievalQuery(text="Mars", top_k=10, principal=principal)
            )
        )
        for name, principal in principals.items()
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, sort_keys=True))
