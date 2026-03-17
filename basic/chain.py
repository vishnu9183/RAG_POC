"""
Basic RAG Chain: The naive implementation.

This chain:
1. Takes user query
2. Retrieves top-k similar chunks (no reranking)
3. Adds FULL conversation history to prompt
4. Sends to LLM

This is what most tutorials show, but it breaks in production.
"""
from typing import Dict, List, Any, Tuple
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from shared.llm import get_llm
from shared.vectorstore import get_vectorstore
from .retriever import get_basic_retriever
from .memory import get_basic_memory, count_memory_tokens


class BasicRAGChain:
    """
    Wrapper around the basic RAG chain with metrics tracking.
    """
    
    def __init__(self):
        self.llm = get_llm()
        self.retriever = get_basic_retriever()
        self.memory = get_basic_memory()
        
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
                    "total_estimated_tokens": int
                }
            }
        """
        # Get metrics before query
        history_tokens = count_memory_tokens(self.memory)
        
        # Run the chain
        result = self.chain.invoke({"question": question})
        
        # Calculate metrics
        source_docs = result.get("source_documents", [])
        chunks_tokens = sum(len(doc.page_content) // 4 for doc in source_docs)
        answer_tokens = len(result["answer"]) // 4
        
        # Estimate total tokens (prompt + response)
        # System prompt ~200 tokens + history + chunks + question + answer
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
                "cumulative_tokens": self.total_tokens_used
            }
        }
    
    def reset_memory(self):
        """Clear conversation history."""
        self.memory.clear()
        self.total_queries = 0
        self.total_tokens_used = 0
    
    def get_retrieved_chunks_with_scores(self, question: str) -> List[Tuple[Document, float]]:
        """
        Get retrieved documents with their similarity scores.
        Useful for debugging and comparison.
        """
        vectorstore = get_vectorstore()
        docs_with_scores = vectorstore.similarity_search_with_score(
            question, 
            k=5
        )
        return docs_with_scores


def create_basic_chain() -> BasicRAGChain:
    """Factory function to create a basic RAG chain."""
    return BasicRAGChain()
