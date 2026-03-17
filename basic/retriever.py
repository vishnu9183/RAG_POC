"""
Basic retriever: Simple top-k semantic search.

This is the naive approach:
- Embed the query
- Find k nearest vectors by cosine similarity
- Return all k results (no filtering or reranking)

Problems this doesn't solve:
1. Keyword queries that don't have semantic matches
2. Irrelevant results that happen to be semantically similar
3. No quality filtering
"""
from langchain_core.retrievers import BaseRetriever
from shared.vectorstore import get_vectorstore
import config


def get_basic_retriever() -> BaseRetriever:
    """
    Get a simple top-k retriever.
    
    Returns the top 5 most similar documents by cosine similarity.
    No reranking, no hybrid search, no filtering.
    """
    vectorstore = get_vectorstore()
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": config.BASIC_TOP_K  # Default: 5
        }
    )
    
    return retriever
