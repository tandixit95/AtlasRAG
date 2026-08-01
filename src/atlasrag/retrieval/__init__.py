"""Framework-independent retrieval contracts and implementations."""

from atlasrag.retrieval.access import (
    GROUPS_METADATA_KEY,
    TENANT_METADATA_KEY,
    PermissionPolicy,
)
from atlasrag.retrieval.bm25 import BM25Retriever, tokenize
from atlasrag.retrieval.contracts import (
    AccessPrincipal,
    RetrievalContribution,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    ScoreKind,
)
from atlasrag.retrieval.dense import ExactDenseRetriever
from atlasrag.retrieval.hybrid import ReciprocalRankFusionRetriever

__all__ = [
    "AccessPrincipal",
    "BM25Retriever",
    "ExactDenseRetriever",
    "GROUPS_METADATA_KEY",
    "PermissionPolicy",
    "ReciprocalRankFusionRetriever",
    "RetrievalContribution",
    "RetrievalMethod",
    "RetrievalQuery",
    "RetrievalResult",
    "Retriever",
    "ScoreKind",
    "TENANT_METADATA_KEY",
    "tokenize",
]
