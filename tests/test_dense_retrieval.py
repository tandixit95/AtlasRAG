from __future__ import annotations
from atlasrag.embeddings.base import EmbeddingModel, Vector
from atlasrag.ingestion.chunking import FixedCharacterChunker
from atlasrag.models import Document
from atlasrag.retrieval.dense import ExactDenseRetriever

class KeywordEmbedding(EmbeddingModel):
    @property
    def model_id(self) -> str: return "keyword-test-v1"
    @property
    def dimension(self) -> int: return 3
    def _embed(self, text: str) -> Vector:
        lowered = text.lower()
        return (float(lowered.count("mars")), float(lowered.count("ocean")), 1.0)
    def embed_documents(self, texts): return tuple(self._embed(text) for text in texts)
    def embed_query(self, text): return self._embed(text)

def _chunks():
    docs = [
        Document.from_text(source_uri="memory://mars", text="mars rover mission"),
        Document.from_text(source_uri="memory://ocean", text="ocean current research"),
        Document.from_text(source_uri="memory://other", text="forest canopy study"),
    ]
    chunker = FixedCharacterChunker(chunk_size=100)
    return tuple(chunker.chunk(doc)[0] for doc in docs)

def test_exact_dense_retrieval_ranks_relevant_chunk_first_and_preserves_provenance():
    chunks = _chunks()
    retriever = ExactDenseRetriever(KeywordEmbedding())
    retriever.index(chunks)
    results = retriever.search("mars", top_k=2)
    assert results[0].chunk.source_uri == "memory://mars"
    assert results[0].rank == 1
    assert results[0].chunk == chunks[0]
    assert results[0].score > results[1].score

def test_top_k_is_capped_by_index_size():
    retriever = ExactDenseRetriever(KeywordEmbedding())
    retriever.index(_chunks())
    assert len(retriever.search("ocean", top_k=20)) == 3

def test_search_before_index_returns_empty_results():
    assert ExactDenseRetriever(KeywordEmbedding()).search("mars") == ()

import pytest

class BrokenCountEmbedding(KeywordEmbedding):
    def embed_documents(self, texts): return ()

class BrokenDimensionEmbedding(KeywordEmbedding):
    def embed_documents(self, texts): return tuple((1.0, 2.0) for _ in texts)

def test_index_rejects_missing_embeddings():
    retriever = ExactDenseRetriever(BrokenCountEmbedding())
    with pytest.raises(ValueError, match="one vector per chunk"):
        retriever.index(_chunks())

def test_index_rejects_wrong_embedding_dimension():
    retriever = ExactDenseRetriever(BrokenDimensionEmbedding())
    with pytest.raises(ValueError, match="dimension"):
        retriever.index(_chunks())

def test_search_rejects_invalid_top_k_and_blank_query():
    retriever = ExactDenseRetriever(KeywordEmbedding())
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("mars", top_k=0)
    with pytest.raises(ValueError, match="blank"):
        retriever.search("   ")

def test_equal_scores_use_chunk_id_as_deterministic_tie_breaker():
    chunks = _chunks()[1:]

    class ConstantEmbedding(KeywordEmbedding):
        def _embed(self, text: str) -> Vector:
            return (1.0, 1.0, 1.0)

    retriever = ExactDenseRetriever(ConstantEmbedding())
    retriever.index(chunks)
    results = retriever.search("anything", top_k=2)
    assert [result.chunk.chunk_id for result in results] == sorted(chunk.chunk_id for chunk in chunks)
