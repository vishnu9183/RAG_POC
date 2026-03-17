"""
RAG Comparison UI - Phase 2 with MCP Integration

Enhanced version that adds:
- Third column showing MCP-enhanced RAG
- Live web data from Brave Search
- Comparison across all three approaches

Run with:
    python app_phase2.py
    
Or with MCP disabled:
    python app_phase2.py --no-mcp
"""
import argparse
import gradio as gr
from typing import List, Tuple
import time

from basic.chain import create_basic_chain
from optimized.chain import create_optimized_chain
from mcp.chain import create_mcp_chain
from shared.vectorstore import get_collection_info


# Global chain instances
basic_chain = None
optimized_chain = None
mcp_chain = None
MCP_ENABLED = True


def initialize_chains(enable_mcp: bool = True):
    """Initialize all RAG chains."""
    global basic_chain, optimized_chain, mcp_chain, MCP_ENABLED
    
    MCP_ENABLED = enable_mcp
    
    print("Initializing RAG chains...")
    basic_chain = create_basic_chain()
    optimized_chain = create_optimized_chain()
    
    if enable_mcp:
        mcp_chain = create_mcp_chain(enable_mcp=True)
        print("MCP chain initialized with Tavily Search.")
    else:
        mcp_chain = None
        print("MCP disabled.")
    
    print("Chains initialized.")


def format_sources(docs, show_scores=False, show_source_type=False) -> str:
    """Format source documents for display."""
    if not docs:
        return "No sources retrieved"
    
    lines = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", doc.metadata.get("filename", "Unknown"))
        
        # Source type indicator
        source_type = ""
        if show_source_type:
            if doc.metadata.get("source_type") == "mcp_live":
                source_type = "🌐 "
                title = doc.metadata.get("title", "Web Result")
            else:
                source_type = "📄 "
        
        # Truncate content
        content_preview = doc.page_content[:150].replace("\n", " ")
        if len(doc.page_content) > 150:
            content_preview += "..."
        
        # Scores
        score_info = ""
        if show_scores and "rerank_score" in doc.metadata:
            score_info = f" [{doc.metadata['rerank_score']:.3f}]"
        
        lines.append(f"**{i}. {source_type}{title}**{score_info}\n{content_preview}\n")
    
    return "\n".join(lines)


def format_metrics(metrics: dict, chain_type: str = "basic") -> str:
    """Format metrics for display."""
    lines = [
        f"📊 **Metrics**",
        f"- History tokens: {metrics['history_tokens']:,}",
        f"- Retrieved chunks: {metrics['retrieved_chunks']}",
        f"- Total tokens: **{metrics['total_estimated_tokens']:,}**",
        f"- Cumulative: {metrics['cumulative_tokens']:,}",
    ]
    
    if chain_type == "optimized" and "memory_breakdown" in metrics:
        mb = metrics["memory_breakdown"]
        lines.append(f"\n💾 **Memory**")
        lines.append(f"- Summary: {mb['summary_tokens']} tokens")
        lines.append(f"- Buffer: {mb['buffer_messages']} msgs")
    
    if chain_type == "mcp":
        lines.append(f"\n🌐 **MCP Data**")
        lines.append(f"- Static results: {metrics.get('static_results', 0)}")
        lines.append(f"- Live results: {metrics.get('mcp_results', 0)}")
        lines.append(f"- MCP in final: {metrics.get('mcp_in_final', 0)}")
        lines.append(f"- Total MCP hits: {metrics.get('total_mcp_hits', 0)}")
    
    return "\n".join(lines)


def chat_phase2(
    message: str,
    basic_history: List[Tuple[str, str]],
    optimized_history: List[Tuple[str, str]],
    mcp_history: List[Tuple[str, str]]
):
    """Process message through all chains."""
    global basic_chain, optimized_chain, mcp_chain
    
    if basic_chain is None:
        initialize_chains(MCP_ENABLED)
    
    # Query basic
    basic_start = time.time()
    basic_result = basic_chain.query(message)
    basic_time = time.time() - basic_start
    
    # Query optimized
    opt_start = time.time()
    opt_result = optimized_chain.query(message)
    opt_time = time.time() - opt_start
    
    # Query MCP (if enabled)
    mcp_result = None
    mcp_time = 0
    if mcp_chain:
        mcp_start = time.time()
        mcp_result = mcp_chain.query(message)
        mcp_time = time.time() - mcp_start
    
    # Update histories
    basic_history = basic_history + [(message, basic_result["answer"])]
    optimized_history = optimized_history + [(message, opt_result["answer"])]
    
    if mcp_result:
        mcp_history = mcp_history + [(message, mcp_result["answer"])]
    
    # Format outputs
    basic_sources = format_sources(basic_result["source_documents"])
    basic_metrics = format_metrics(basic_result["metrics"], "basic")
    basic_metrics += f"\n- Latency: {basic_time:.2f}s"
    
    opt_sources = format_sources(opt_result["source_documents"], show_scores=True)
    opt_metrics = format_metrics(opt_result["metrics"], "optimized")
    opt_metrics += f"\n- Latency: {opt_time:.2f}s"
    
    if mcp_result:
        mcp_sources = format_sources(
            mcp_result["source_documents"], 
            show_scores=True, 
            show_source_type=True
        )
        mcp_metrics = format_metrics(mcp_result["metrics"], "mcp")
        mcp_metrics += f"\n- Latency: {mcp_time:.2f}s"
    else:
        mcp_sources = "MCP disabled"
        mcp_metrics = "MCP disabled"
    
    # Comparison table
    basic_tokens = basic_result["metrics"]["total_estimated_tokens"]
    opt_tokens = opt_result["metrics"]["total_estimated_tokens"]
    mcp_tokens = mcp_result["metrics"]["total_estimated_tokens"] if mcp_result else 0
    
    comparison = f"""### 📊 Comparison

| Metric | Basic | Optimized | MCP |
|--------|-------|-----------|-----|
| Tokens | {basic_tokens:,} | {opt_tokens:,} | {mcp_tokens:,} |
| Latency | {basic_time:.2f}s | {opt_time:.2f}s | {mcp_time:.2f}s |
| Chunks | {basic_result['metrics']['retrieved_chunks']} | {opt_result['metrics']['retrieved_chunks']} | {mcp_result['metrics']['retrieved_chunks'] if mcp_result else '-'} |
"""
    
    if mcp_result and mcp_result["metrics"].get("mcp_in_final", 0) > 0:
        comparison += f"\n🌐 **Live web data contributed to this answer!**"
    
    return (
        basic_history,
        optimized_history,
        mcp_history,
        basic_sources,
        basic_metrics,
        opt_sources,
        opt_metrics,
        mcp_sources,
        mcp_metrics,
        comparison,
        ""
    )


def reset_all():
    """Reset all chains."""
    global basic_chain, optimized_chain, mcp_chain
    
    if basic_chain:
        basic_chain.reset_memory()
    if optimized_chain:
        optimized_chain.reset_memory()
    if mcp_chain:
        mcp_chain.reset_memory()
    
    return [], [], [], "", "", "", "", "", "", "Conversation reset!", ""


def create_phase2_ui():
    """Create the Phase 2 Gradio interface."""
    
    collection_info = get_collection_info()
    if "error" in collection_info:
        status_msg = f"⚠️ Qdrant: {collection_info['error']}"
    else:
        status_msg = f"✅ Qdrant: {collection_info['points_count']} docs"
    
    mcp_status = "✅ MCP: Tavily Search enabled" if MCP_ENABLED else "⚠️ MCP: Disabled"
    
    with gr.Blocks(
        title="RAG Comparison - Phase 2",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown(
            f"""
            # 🔬 RAG Pipeline Comparison - Phase 2
            
            Compare **Basic** vs **Optimized** vs **MCP-Enhanced** RAG
            
            *{status_msg} | {mcp_status}*
            """
        )
        
        with gr.Row():
            # Basic RAG
            with gr.Column(scale=1):
                gr.Markdown("## 🔵 Basic RAG")
                basic_chat = gr.Chatbot(height=300, type="tuples")
                with gr.Accordion("Sources", open=False):
                    basic_src = gr.Markdown("")
                with gr.Accordion("Metrics", open=True):
                    basic_met = gr.Markdown("*Send a message*")
            
            # Optimized RAG
            with gr.Column(scale=1):
                gr.Markdown("## 🟢 Optimized RAG")
                opt_chat = gr.Chatbot(height=300, type="tuples")
                with gr.Accordion("Sources", open=False):
                    opt_src = gr.Markdown("")
                with gr.Accordion("Metrics", open=True):
                    opt_met = gr.Markdown("*Send a message*")
            
            # MCP RAG
            with gr.Column(scale=1):
                gr.Markdown("## 🌐 MCP + Optimized")
                mcp_chat = gr.Chatbot(height=300, type="tuples")
                with gr.Accordion("Sources", open=False):
                    mcp_src = gr.Markdown("")
                with gr.Accordion("Metrics", open=True):
                    mcp_met = gr.Markdown("*Send a message*")
        
        comparison = gr.Markdown("")
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask a question...",
                show_label=False,
                scale=4
            )
            send = gr.Button("Send", variant="primary")
            reset = gr.Button("🔄 Reset")
        
        gr.Markdown("### 💡 Try these:")
        gr.Examples(
            examples=[
                "What does Paul Graham say about startups?",
                "What are the latest AI trends?",  # MCP will help here
                "How should founders approach hiring?",
                "What essay mentions Lisp?",
                "What's happening in tech today?",  # MCP will help here
            ],
            inputs=msg
        )
        
        # Events
        inputs = [msg, basic_chat, opt_chat, mcp_chat]
        outputs = [
            basic_chat, opt_chat, mcp_chat,
            basic_src, basic_met,
            opt_src, opt_met,
            mcp_src, mcp_met,
            comparison, msg
        ]
        
        send.click(fn=chat_phase2, inputs=inputs, outputs=outputs)
        msg.submit(fn=chat_phase2, inputs=inputs, outputs=outputs)
        reset.click(fn=reset_all, outputs=outputs)
        
        gr.Markdown(
            """
            ---
            ### 🔍 What to observe:
            
            1. **Static vs Live**: Ask about "latest AI trends" - MCP pulls live web data
            2. **Token Growth**: Watch Basic RAG explode while others stay bounded
            3. **Source Types**: MCP results show 🌐 for web, 📄 for static corpus
            4. **Relevance**: Reranking filters irrelevant results in Optimized/MCP
            """
        )
    
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mcp", action="store_true", help="Disable MCP features")
    args = parser.parse_args()
    
    enable_mcp = not args.no_mcp
    
    print("Starting Phase 2 RAG Comparison...")
    initialize_chains(enable_mcp=enable_mcp)
    
    app = create_phase2_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
