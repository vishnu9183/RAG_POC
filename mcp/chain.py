"""
MCP-Enhanced RAG Chain: Optimized RAG + Live web data via Tavily.

This chain combines:
1. Static knowledge from Paul Graham essays (hybrid search + reranking)
2. Live web data from Tavily Search API

This demonstrates how live data can augment a static RAG pipeline —
useful for queries about recent events that the essay corpus doesn't cover.
"""
from typing import Dict, List, Any
from langchain_core.documents import Document
from optimized.retriever import get_optimized_retriever
from optimized.memory import get_optimized_memory, count_memory_tokens, get_memory_breakdown
from shared.llm import get_llm
import config


_tavily = None


def get_tavily():
    """Load the Tavily search tool (lazy singleton)."""
    global _tavily
    if _tavily is None:
        from langchain_community.tools.tavily_search import TavilySearchResults
        _tavily = TavilySearchResults(
            api_key=config.TAVILY_API_KEY,
            max_results=3
        )
    return _tavily


class MCPChain:
    """
    RAG chain that merges static essay retrieval with live Tavily web search.

    Flow:
      Query → Hybrid search (essays) → Rerank → top 3 static docs
            → Tavily search          → top 3 live docs
            → Combine → LLM with summarized memory → Answer
    """

    def __init__(self, enable_mcp: bool = True):
        self.llm = get_llm()
        self.retriever = get_optimized_retriever()
        self.memory = get_optimized_memory()
        self.enable_mcp = enable_mcp

        self.total_queries = 0
        self.total_tokens_used = 0
        self.total_mcp_hits = 0

    def query(self, question: str) -> Dict[str, Any]:
        history_tokens = count_memory_tokens(self.memory)
        memory_breakdown = get_memory_breakdown(self.memory)

        # Step 1: Static retrieval (hybrid search + reranking)
        static_docs = self.retriever._get_relevant_documents(question)

        # Step 2: Live web search via Tavily
        mcp_docs = []
        if self.enable_mcp and config.TAVILY_API_KEY:
            try:
                results = get_tavily().run(question)
                if isinstance(results, list):
                    for r in results:
                        mcp_docs.append(Document(
                            page_content=r.get("content", ""),
                            metadata={
                                "title": r.get("title", "Web Result"),
                                "url": r.get("url", ""),
                                "source_type": "mcp_live"
                            }
                        ))
            except Exception as e:
                print(f"Tavily search error: {e}")

        # Step 3: Combine static + live docs
        all_docs = static_docs + mcp_docs

        # Step 4: Build context string
        context_parts = []
        for i, doc in enumerate(all_docs, 1):
            label = "🌐 Web" if doc.metadata.get("source_type") == "mcp_live" else "📄 Essay"
            context_parts.append(f"[{i}] ({label})\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        # Step 5: Load conversation history
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        history_str = ""
        if chat_history:
            if isinstance(chat_history, str):
                history_str = chat_history
            else:
                history_str = "\n".join([f"{m.type}: {m.content}" for m in chat_history])

        # Step 6: Build prompt
        history_section = f"\nPrevious conversation:\n{history_str}" if history_str else ""
        prompt = (
            "You are a helpful assistant with access to Paul Graham's essays and live web data.\n"
            "Use the provided context to answer accurately. Cite sources where relevant.\n\n"
            f"Context:\n{context}"
            f"{history_section}\n\n"
            f"Question: {question}\nAnswer:"
        )

        # Step 7: Invoke LLM
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # Step 8: Save to memory
        self.memory.save_context({"input": question}, {"answer": answer})

        # Metrics
        mcp_in_final = len(mcp_docs)
        self.total_mcp_hits += mcp_in_final
        chunks_tokens = sum(len(doc.page_content) // 4 for doc in all_docs)
        answer_tokens = len(answer) // 4
        total_tokens = 200 + history_tokens + chunks_tokens + len(question) // 4 + answer_tokens
        self.total_queries += 1
        self.total_tokens_used += total_tokens

        return {
            "answer": answer,
            "source_documents": all_docs,
            "metrics": {
                "history_tokens": history_tokens,
                "retrieved_chunks": len(all_docs),
                "chunks_tokens": chunks_tokens,
                "answer_tokens": answer_tokens,
                "total_estimated_tokens": total_tokens,
                "cumulative_tokens": self.total_tokens_used,
                "memory_breakdown": memory_breakdown,
                "static_results": len(static_docs),
                "mcp_results": len(mcp_docs),
                "mcp_in_final": mcp_in_final,
                "total_mcp_hits": self.total_mcp_hits
            }
        }

    def reset_memory(self):
        """Clear conversation history and metrics."""
        self.memory.clear()
        self.total_queries = 0
        self.total_tokens_used = 0
        self.total_mcp_hits = 0


def create_mcp_chain(enable_mcp: bool = True) -> MCPChain:
    """Factory function to create the MCP-enhanced chain."""
    return MCPChain(enable_mcp=enable_mcp)
