"""
Optimized memory: Conversation Summary Buffer.

This is the production approach to conversation memory:

1. Keep recent messages verbatim (last 3-4 turns)
2. Summarize older messages into a compressed form
3. Total token usage stays bounded

How it works:
- When history exceeds max_token_limit, older messages get summarized
- Summary is prepended to the conversation
- New messages are added normally
- Cycle repeats when limit is hit again

Benefits:
- Constant memory usage regardless of conversation length
- Recent context preserved exactly
- Older context preserved in compressed form
- No token explosion
"""
from langchain.memory import ConversationSummaryBufferMemory
from shared.llm import get_summary_llm
import config


def get_optimized_memory() -> ConversationSummaryBufferMemory:
    """
    Get a summary buffer memory that keeps tokens bounded.
    
    When conversation exceeds max_token_limit:
    - Older messages are summarized
    - Summary replaces the old messages
    - Total stays under the limit
    """
    llm = get_summary_llm()  # Uses lower temperature for consistent summaries
    
    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=config.SUMMARY_BUFFER_MAX_TOKENS,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )


def count_memory_tokens(memory: ConversationSummaryBufferMemory) -> int:
    """
    Estimate tokens in the conversation history.
    Includes both the summary and the buffer.
    """
    # Get the moving summary
    summary_tokens = len(memory.moving_summary_buffer) // 4 if memory.moving_summary_buffer else 0
    
    # Get the buffer messages
    buffer_tokens = sum(len(m.content) // 4 for m in memory.chat_memory.messages)
    
    return summary_tokens + buffer_tokens


def get_memory_breakdown(memory: ConversationSummaryBufferMemory) -> dict:
    """
    Get detailed breakdown of memory usage.
    """
    summary = memory.moving_summary_buffer or ""
    messages = memory.chat_memory.messages
    
    return {
        "summary_tokens": len(summary) // 4,
        "summary_preview": summary[:200] + "..." if len(summary) > 200 else summary,
        "buffer_messages": len(messages),
        "buffer_tokens": sum(len(m.content) // 4 for m in messages),
        "total_tokens": len(summary) // 4 + sum(len(m.content) // 4 for m in messages)
    }
