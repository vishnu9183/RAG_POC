"""
Embedding model setup using HuggingFace BGE.
"""
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import config


_embedding_model = None


def get_embedding_model() -> HuggingFaceBgeEmbeddings:
    """Load the BGE embedding model (lazy, singleton)."""
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedding_model = HuggingFaceBgeEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("Embedding model loaded.")
    return _embedding_model


def embed_texts(texts: list) -> list:
    """Embed a list of texts (for ingestion)."""
    return get_embedding_model().embed_documents(texts)


def embed_query(query: str) -> list:
    """Embed a single query string."""
    return get_embedding_model().embed_query(query)
