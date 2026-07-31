"""Sentence Transformers embedding adapter."""
from __future__ import annotations
from collections.abc import Sequence
from atlasrag.embeddings.base import EmbeddingModel, Vector

class SentenceTransformerEmbedding(EmbeddingModel):
    """Lazy adapter around a Sentence Transformers model.

    The optional dependency is loaded only when this adapter is instantiated,
    keeping AtlasRAG's core domain and deterministic tests lightweight.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "SentenceTransformerEmbedding requires the 'embeddings' extra"
            ) from exc
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        if not texts:
            return ()
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return tuple(tuple(float(value) for value in vector) for vector in vectors)

    def embed_query(self, text: str) -> Vector:
        if not text.strip():
            raise ValueError("query text must not be blank")
        vector = self._model.encode(text, normalize_embeddings=True)
        return tuple(float(value) for value in vector)
