"""
Qdrant vector store setup with support for hybrid search.
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from langchain_community.vectorstores import Qdrant
from typing import Optional
import config
from .embeddings import get_embedding_model


# Global client instance
_qdrant_client = None


def get_qdrant_client() -> QdrantClient:
    """
    Get the Qdrant client instance.
    """
    global _qdrant_client
    
    if _qdrant_client is None:
        print(f"Connecting to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
        _qdrant_client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )
        print("Connected to Qdrant.")
    
    return _qdrant_client


def create_collection(recreate: bool = False):
    """
    Create the Qdrant collection with both dense and sparse vector support.
    
    Dense vectors: BGE embeddings for semantic search
    Sparse vectors: BM25 for keyword matching (hybrid search)
    """
    client = get_qdrant_client()
    
    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == config.COLLECTION_NAME for c in collections)
    
    if exists and not recreate:
        print(f"Collection '{config.COLLECTION_NAME}' already exists.")
        return
    
    if exists and recreate:
        print(f"Deleting existing collection '{config.COLLECTION_NAME}'...")
        client.delete_collection(config.COLLECTION_NAME)
    
    # Create collection with hybrid vector support
    print(f"Creating collection '{config.COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=config.EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )
    print("Collection created with dense + sparse vector support.")


def get_vectorstore() -> Qdrant:
    """
    Get the LangChain Qdrant vectorstore wrapper.
    Used for basic retrieval operations.
    """
    embeddings = get_embedding_model()
    client = get_qdrant_client()
    
    return Qdrant(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embeddings=embeddings,
        vector_name="dense"  # Use the dense vectors
    )


def get_collection_info() -> dict:
    """
    Get information about the collection.
    """
    client = get_qdrant_client()
    try:
        info = client.get_collection(config.COLLECTION_NAME)
        return {
            "name": config.COLLECTION_NAME,
            "points_count": info.points_count,
            "status": info.status
        }
    except Exception as e:
        return {"error": str(e)}
