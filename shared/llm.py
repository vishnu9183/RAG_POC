"""
LLM setup using Groq API (Llama 3.1 70B)
"""
from langchain_groq import ChatGroq
import config


def get_llm() -> ChatGroq:
    """
    Get the Groq LLM instance.
    Uses Llama 3.1 70B which is free and fast.
    """
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model_name=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS
    )


def get_summary_llm() -> ChatGroq:
    """
    Get a separate LLM instance for summarization.
    Uses lower temperature for more consistent summaries.
    """
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model_name=config.LLM_MODEL,
        temperature=0.3,  # Lower for summarization
        max_tokens=512
    )
