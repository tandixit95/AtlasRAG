"""Embedding interfaces and adapters."""

from atlasrag.embeddings.base import EmbeddingModel, Vector
from atlasrag.embeddings.sentence_transformers import SentenceTransformerEmbedding

__all__ = ["EmbeddingModel", "SentenceTransformerEmbedding", "Vector"]
