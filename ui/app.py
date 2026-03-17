"""
RAG Comparison UI

A side-by-side Gradio interface comparing Basic RAG vs Optimized RAG.

Features:
- Split view with both pipelines
- Real-time token tracking
- Retrieved chunks visualization
- Cumulative metrics over conversation

Run with:
    python app.py
    
Then open http://localhost:7860
"""
import gradio as gr
from typing import List
import time
import json
import os
from datetime import datetime

from basic.chain import create_basic_chain
from optimized.chain import create_optimized_chain
from shared.vectorstore import get_collection_info


# Global chain instances
basic_chain = None
optimized_chain = None

# Log file path
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "metrics.jsonl")


def log_metrics(question: str, basic_metrics: dict, opt_metrics: dict, basic_error: str = None, opt_error: str = None):
    """Append a metrics record to logs/metrics.jsonl."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "basic": basic_metrics if not basic_error else {"error": basic_error},
        "optimized": opt_metrics if not opt_error else {"error": opt_error},
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def initialize_chains():
    """Initialize both RAG chains."""
    global basic_chain, optimized_chain
    
    print("Initializing RAG chains...")
    basic_chain = create_basic_chain()
    optimized_chain = create_optimized_chain()
    print("Chains initialized.")


def format_sources(docs, show_scores=False) -> str:
    """Format source documents for display."""
    if not docs:
        return "No sources retrieved"
    
    lines = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", doc.metadata.get("filename", "Unknown"))
        
        # Truncate content for display
        content_preview = doc.page_content[:150].replace("\n", " ")
        if len(doc.page_content) > 150:
            content_preview += "..."
        
        # Add scores if available
        score_info = ""
        if show_scores:
            if "rerank_score" in doc.metadata:
                score_info = f" [rerank: {doc.metadata['rerank_score']:.3f}]"
            elif "initial_score" in doc.metadata:
                score_info = f" [score: {doc.metadata['initial_score']:.3f}]"
        
        lines.append(f"**{i}. {title}**{score_info}\n{content_preview}\n")
    
    return "\n".join(lines)


def format_metrics(metrics: dict, is_optimized: bool = False) -> str:
    """Format metrics for display."""
    if not metrics:
        return "📊 **Metrics**\n- N/A (chain error)"
    lines = [
        f"📊 **Metrics**",
        f"- History tokens: {metrics.get('history_tokens', 0):,}",
        f"- Retrieved chunks: {metrics.get('retrieved_chunks', 0)}",
        f"- Chunk tokens: {metrics.get('chunks_tokens', 0):,}",
        f"- Answer tokens: {metrics.get('answer_tokens', 0):,}",
        f"- **Total tokens: {metrics.get('total_estimated_tokens', 0):,}**",
        f"- Cumulative: {metrics.get('cumulative_tokens', 0):,}",
    ]
    
    if is_optimized and "memory_breakdown" in metrics:
        mb = metrics["memory_breakdown"]
        lines.append(f"\n💾 **Memory Breakdown**")
        lines.append(f"- Summary tokens: {mb['summary_tokens']}")
        lines.append(f"- Buffer messages: {mb['buffer_messages']}")
        lines.append(f"- Buffer tokens: {mb['buffer_tokens']}")
    
    return "\n".join(lines)


def chat(
    message: str,
    basic_history: List[dict],
    optimized_history: List[dict]
):
    """
    Process a message through both chains.
    
    Returns updated histories, metrics, and cleared input.
    Must return exactly 8 values to match outputs list.
    """
    global basic_chain, optimized_chain
    
    if basic_chain is None or optimized_chain is None:
        initialize_chains()
    
    # Query basic chain (isolated — errors don't stop optimized chain)
    basic_error = None
    basic_start = time.time()
    try:
        basic_result = basic_chain.query(message)
    except Exception as e:
        basic_error = str(e)
        basic_result = {"answer": f"⚠️ Error: {e}", "source_documents": [], "metrics": {}}
    basic_time = time.time() - basic_start

    # Query optimized chain (isolated — errors don't stop basic chain)
    opt_error = None
    opt_start = time.time()
    try:
        opt_result = optimized_chain.query(message)
    except Exception as e:
        opt_error = str(e)
        opt_result = {"answer": f"⚠️ Error: {e}", "source_documents": [], "metrics": {}}
    opt_time = time.time() - opt_start

    # Log metrics to file
    log_metrics(message, basic_result["metrics"], opt_result["metrics"], basic_error, opt_error)

    # Update histories
    basic_history = basic_history + [{"role": "user", "content": message}, {"role": "assistant", "content": basic_result["answer"]}]
    optimized_history = optimized_history + [{"role": "user", "content": message}, {"role": "assistant", "content": opt_result["answer"]}]

    # Format outputs
    basic_sources = format_sources(basic_result["source_documents"])
    basic_metrics = format_metrics(basic_result["metrics"])
    basic_metrics += f"\n- Latency: {basic_time:.2f}s"

    opt_sources = format_sources(opt_result["source_documents"], show_scores=True)
    opt_metrics = format_metrics(opt_result["metrics"], is_optimized=True)
    opt_metrics += f"\n- Latency: {opt_time:.2f}s"

    # Calculate comparison
    basic_tokens = basic_result["metrics"].get("total_estimated_tokens", 0)
    opt_tokens = opt_result["metrics"].get("total_estimated_tokens", 0)
    savings = ((basic_tokens - opt_tokens) / basic_tokens * 100) if basic_tokens > 0 else 0

    comparison = f"""### Comparison
| Metric | Basic | Optimized | Δ |
|--------|-------|-----------|---|
| Total Tokens | {basic_tokens:,} | {opt_tokens:,} | **{savings:.1f}% saved** |
| Latency | {basic_time:.2f}s | {opt_time:.2f}s | {((basic_time-opt_time)/basic_time*100):.1f}% |
| Chunks Retrieved | {basic_result['metrics'].get('retrieved_chunks', 'N/A')} | {opt_result['metrics'].get('retrieved_chunks', 'N/A')} | - |
"""
    
    # Return exactly 8 values matching the outputs list
    return (
        basic_history,           # 1. basic_chatbot
        optimized_history,       # 2. optimized_chatbot
        basic_sources,           # 3. basic_sources_display
        basic_metrics,           # 4. basic_metrics_display
        opt_sources,             # 5. opt_sources_display
        opt_metrics,             # 6. opt_metrics_display
        comparison,              # 7. comparison_display
        ""                       # 8. msg_input (cleared)
    )


def reset_conversation():
    """Reset both chains' memory."""
    global basic_chain, optimized_chain
    
    if basic_chain:
        basic_chain.reset_memory()
    if optimized_chain:
        optimized_chain.reset_memory()
    
    # Return exactly 8 values matching the outputs list
    return (
        [],                              # 1. basic_chatbot
        [],                              # 2. optimized_chatbot
        "",                              # 3. basic_sources_display
        "",                              # 4. basic_metrics_display
        "",                              # 5. opt_sources_display
        "",                              # 6. opt_metrics_display
        "Conversation reset. Start fresh!",  # 7. comparison_display
        ""                               # 8. msg_input (cleared)
    )


def create_ui():
    """Create the Gradio interface."""
    
    # Check collection status
    collection_info = get_collection_info()
    if "error" in collection_info:
        status_msg = f"⚠️ Qdrant not ready: {collection_info['error']}"
    else:
        status_msg = f"✅ Connected to Qdrant | Collection: {collection_info['name']} | Documents: {collection_info['points_count']}"
    
    with gr.Blocks(title="RAG Comparison") as app:
        
        gr.Markdown(
            """
            # 🔬 RAG Pipeline Comparison
            
            Compare **Basic RAG** (naive implementation) vs **Optimized RAG** (production-grade).
            
            Watch the token counts diverge as you chat longer!
            """
        )
        
        gr.Markdown(f"*{status_msg}*")
        
        with gr.Row():
            # Left panel - Basic RAG
            with gr.Column():
                gr.Markdown("## 🔵 Basic RAG")
                gr.Markdown("*Top-K retrieval • Full history • No reranking*")
                
                basic_chatbot = gr.Chatbot(
                    label="Basic RAG Chat",
                    height=350
                )
                
                with gr.Accordion("Retrieved Chunks", open=False):
                    basic_sources_display = gr.Markdown("")
                
                with gr.Accordion("Metrics", open=True):
                    basic_metrics_display = gr.Markdown("*Send a message to see metrics*")
            
            # Right panel - Optimized RAG
            with gr.Column():
                gr.Markdown("## 🟢 Optimized RAG")
                gr.Markdown("*Hybrid search • Reranking • Summary memory*")
                
                optimized_chatbot = gr.Chatbot(
                    label="Optimized RAG Chat",
                    height=350
                )
                
                with gr.Accordion("Retrieved Chunks (with rerank scores)", open=False):
                    opt_sources_display = gr.Markdown("")
                
                with gr.Accordion("Metrics", open=True):
                    opt_metrics_display = gr.Markdown("*Send a message to see metrics*")
        
        # Comparison summary
        comparison_display = gr.Markdown("")
        
        # Input area
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask a question about Paul Graham's essays...",
                label="Your Question",
                scale=4,
                show_label=False
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
            reset_btn = gr.Button("🔄 Reset", scale=1)
        
        # Example queries
        gr.Markdown("### 💡 Try these queries:")
        with gr.Row():
            gr.Examples(
                examples=[
                    "What does Paul Graham say about startups?",
                    "How should I approach hiring engineers?",
                    "What essay mentions Lisp?",
                    "Tell me about Y Combinator",
                    "What mistakes do founders make?",
                ],
                inputs=msg_input,
                label=""
            )
        
        # Define outputs list once - must match return values exactly
        all_outputs = [
            basic_chatbot,           # 1
            optimized_chatbot,       # 2
            basic_sources_display,   # 3
            basic_metrics_display,   # 4
            opt_sources_display,     # 5
            opt_metrics_display,     # 6
            comparison_display,      # 7
            msg_input                # 8
        ]
        
        # Event handlers - no .then() chains needed
        send_btn.click(
            fn=chat,
            inputs=[msg_input, basic_chatbot, optimized_chatbot],
            outputs=all_outputs
        )
        
        msg_input.submit(
            fn=chat,
            inputs=[msg_input, basic_chatbot, optimized_chatbot],
            outputs=all_outputs
        )
        
        reset_btn.click(
            fn=reset_conversation,
            inputs=[],
            outputs=all_outputs
        )
        
        gr.Markdown(
            """
            ---
            ### 📚 What to observe:
            
            1. **Token Growth**: Watch Basic RAG's "History tokens" climb with each turn while Optimized stays bounded
            2. **Retrieval Quality**: Compare the chunks retrieved - Optimized should be more relevant after reranking
            3. **Keyword Queries**: Try "What essay mentions Lisp?" - Optimized's hybrid search handles this better
            4. **After 10+ turns**: Basic RAG will slow down or fail; Optimized continues working
            
            ### 🛠️ Architecture:
            
            | Component | Basic | Optimized |
            |-----------|-------|-----------|
            | Search | Dense only (cosine) | Hybrid (dense + BM25) |
            | Filtering | None | Cross-encoder reranker |
            | Memory | Full buffer | Summary + recent buffer |
            | Retrieved | Top 5 | Top 10 → Rerank → Top 3 |
            """
        )
    
    return app


if __name__ == "__main__":
    print("Starting RAG Comparison UI...")
    print("Make sure Qdrant is running: docker ps | grep qdrant")
    print("Make sure documents are ingested: python ingest.py")
    print()
    
    # Initialize chains on startup
    initialize_chains()
    
    # Create and launch UI
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )