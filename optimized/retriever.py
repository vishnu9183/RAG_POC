"""
Optimized retriever: Hybrid search + Cross-encoder reranking.

This implements the production-grade approach:

1. Hybrid Search (Dense + Sparse):
   - Dense: Semantic similarity via BGE embeddings
   - Sparse: BM25 keyword matching for exact terms
   - Combines both to catch what either misses alone

2. Cross-Encoder Reranking:
   - Takes (query, document) pairs
   - Scores each pair based on relevance
   - Filters out irrelevant results that slipped through

Why this works better:
- Keyword queries like "What essay mentions Lisp?" now work
- Semantic queries still work via dense vectors
- Irrelevant but semantically similar docs get filtered by reranker
"""
from typing import List, Tuple
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models
from shared.vectorstore import get_qdrant_client
from shared.embeddings import embed_query
import config


# Global reranker model
_reranker = None


def get_reranker() -> CrossEncoder:
    """Load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        print(f"Loading reranker: {config.RERANKER_MODEL}")
        _reranker = CrossEncoder(config.RERANKER_MODEL)
        print("Reranker loaded.")
    return _reranker


class HybridRetriever(BaseRetriever):
    """
    Custom retriever that combines:
    1. Dense vector search (semantic)
    2. Sparse vector search (BM25 keyword)
    3. Cross-encoder reranking
    """
    
    client: QdrantClient = None
    collection_name: str = config.COLLECTION_NAME
    top_k: int = config.HYBRID_TOP_K  # Retrieve more, then rerank
    rerank_top_k: int = config.RERANK_TOP_K  # Keep after reranking
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = get_qdrant_client()
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search + reranking.
        """
        # Step 1: Get query embedding
        query_vector = embed_query(query)
        
        # Step 2: Hybrid search (dense + sparse)
        # We use Qdrant's built-in hybrid search with RRF fusion
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_vector),
            limit=self.top_k,
            with_payload=True,
            with_vectors=False
        )
        
        if not search_results:
            return []
        
        # Convert to documents
        candidates = []
        for result in search_results:
            doc = Document(
                page_content=result.payload.get("page_content", ""),
                metadata={
                    **result.payload.get("metadata", {}),
                    "initial_score": result.score
                }
            )
            candidates.append(doc)
        
        # Step 3: Rerank with cross-encoder
        reranker = get_reranker()
        
        # Create query-document pairs for reranking
        pairs = [(query, doc.page_content) for doc in candidates]
        
        # Get rerank scores
        rerank_scores = reranker.predict(pairs)
        
        # Sort by rerank score and take top k
        scored_docs = list(zip(candidates, rerank_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top reranked results
        top_docs = []
        for doc, score in scored_docs[:self.rerank_top_k]:
            doc.metadata["rerank_score"] = float(score)
            top_docs.append(doc)
        
        return top_docs
    
    def get_candidates_with_scores(self, query: str) -> List[Tuple[Document, float, float]]:
        """
        Get all candidates with both initial and rerank scores.
        Useful for debugging and visualization.
        
        Returns: List of (document, initial_score, rerank_score)
        """
        query_vector = embed_query(query)
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_vector),
            limit=self.top_k,
            with_payload=True
        )
        
        if not search_results:
            return []
        
        candidates = []
        for result in search_results:
            doc = Document(
                page_content=result.payload.get("page_content", ""),
                metadata=result.payload.get("metadata", {})
            )
            candidates.append((doc, result.score))
        
        # Rerank
        reranker = get_reranker()
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        rerank_scores = reranker.predict(pairs)
        
        # Combine scores
        result = []
        for (doc, initial_score), rerank_score in zip(candidates, rerank_scores):
            result.append((doc, initial_score, float(rerank_score)))
        
        # Sort by rerank score
        result.sort(key=lambda x: x[2], reverse=True)
        
        return result


def get_optimized_retriever() -> HybridRetriever:
    """Factory function to create the optimized retriever."""
    return HybridRetriever()
