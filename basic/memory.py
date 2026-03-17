"""
Basic memory: Full conversation buffer.

This stores EVERY message in the conversation history.
It's the naive approach that causes token explosion.

Problems:
1. Token count grows linearly with conversation length
2. After ~10-15 turns, you hit context limits
3. Older context becomes less relevant but still uses tokens
4. No summarization = wasted tokens on irrelevant history
"""
from langchain.memory import ConversationBufferMemory


def get_basic_memory() -> ConversationBufferMemory:
    """
    Get a conversation buffer that stores full history.
    
    This is intentionally the "wrong" approach to demonstrate
    why production systems need summarization.
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )


def count_memory_tokens(memory: ConversationBufferMemory) -> int:
    """
    Estimate tokens in the conversation history.
    """
    messages = memory.chat_memory.messages
    total_chars = sum(len(m.content) for m in messages)
    return total_chars // 4  # Rough estimate
