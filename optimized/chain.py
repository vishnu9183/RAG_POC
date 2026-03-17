"""
Optimized RAG Chain: Production-grade implementation.

This chain:
1. Takes user query
2. Retrieves via hybrid search (dense + sparse)
3. Reranks results with cross-encoder
4. Adds SUMMARIZED conversation history
5. Sends to LLM

Key differences from basic:
- Hybrid search catches keyword queries
- Reranker filters irrelevant results
- Summary memory keeps tokens bounded
"""
from typing import Dict, List, Any, Tuple
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from shared.llm import get_llm
from .retriever import get_optimized_retriever, HybridRetriever
from .memory import get_optimized_memory, count_memory_tokens, get_memory_breakdown


class OptimizedRAGChain:
    """
    Wrapper around the optimized RAG chain with metrics tracking.
    """
    
    def __init__(self):
        self.llm = get_llm()
        self.retriever = get_optimized_retriever()
        self.memory = get_optimized_memory()
        
        # Track metrics
        self.total_queries = 0
        self.total_tokens_used = 0
        
        # Build the chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query and return response with metrics.
        
        Returns:
            {
                "answer": str,
                "source_documents": List[Document],
                "metrics": {
                    "history_tokens": int,
                    "retrieved_chunks": int,
                    "total_estimated_tokens": int,
                    "memory_breakdown": dict
                }
            }
        """
        # Get metrics before query
        history_tokens = count_memory_tokens(self.memory)
        memory_breakdown = get_memory_breakdown(self.memory)
        
        # Run the chain
        result = self.chain.invoke({"question": question})
        
        # Calculate metrics
        source_docs = result.get("source_documents", [])
        chunks_tokens = sum(len(doc.page_content) // 4 for doc in source_docs)
        answer_tokens = len(result["answer"]) // 4
        
        # Estimate total tokens
        total_tokens = 200 + history_tokens + chunks_tokens + len(question)//4 + answer_tokens
        
        self.total_queries += 1
        self.total_tokens_used += total_tokens
        
        return {
            "answer": result["answer"],
            "source_documents": source_docs,
            "metrics": {
                "history_tokens": history_tokens,
                "retrieved_chunks": len(source_docs),
                "chunks_tokens": chunks_tokens,
                "answer_tokens": answer_tokens,
                "total_estimated_tokens": total_tokens,
                "cumulative_tokens": self.total_tokens_used,
                "memory_breakdown": memory_breakdown
            }
        }
    
    def reset_memory(self):
        """Clear conversation history."""
        self.memory.clear()
        self.total_queries = 0
        self.total_tokens_used = 0
    
    def get_retrieval_details(self, question: str) -> List[Tuple[Document, float, float]]:
        """
        Get detailed retrieval info with both initial and rerank scores.
        
        Returns: List of (document, initial_score, rerank_score)
        """
        if isinstance(self.retriever, HybridRetriever):
            return self.retriever.get_candidates_with_scores(question)
        return []


def create_optimized_chain() -> OptimizedRAGChain:
    """Factory function to create an optimized RAG chain."""
    return OptimizedRAGChain()
