"""Framework-independent retrieval contracts and implementations."""

from atlasrag.retrieval.access import (
    GROUPS_METADATA_KEY,
    TENANT_METADATA_KEY,
    PermissionPolicy,
)
from atlasrag.retrieval.bm25 import BM25Retriever, tokenize
from atlasrag.retrieval.contracts import (
    AccessPrincipal,
    Citation,
    RerankTrace,
    RetrievalContribution,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    ScoreKind,
)
from atlasrag.retrieval.dense import ExactDenseRetriever
from atlasrag.retrieval.hybrid import ReciprocalRankFusionRetriever
from atlasrag.retrieval.reranking import (
    CrossEncoderReranker,
    RerankedRetriever,
    Reranker,
)

__all__ = [
    "AccessPrincipal",
    "BM25Retriever",
    "Citation",
    "CrossEncoderReranker",
    "ExactDenseRetriever",
    "GROUPS_METADATA_KEY",
    "PermissionPolicy",
    "ReciprocalRankFusionRetriever",
    "RerankedRetriever",
    "Reranker",
    "RerankTrace",
    "RetrievalContribution",
    "RetrievalMethod",
    "RetrievalQuery",
    "RetrievalResult",
    "Retriever",
    "ScoreKind",
    "TENANT_METADATA_KEY",
    "tokenize",
]
