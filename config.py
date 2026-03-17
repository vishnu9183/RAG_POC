"""
Configuration settings for RAG Comparison POC
"""
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API Keys
# =============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Validate required keys
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

# =============================================================================
# Qdrant Settings
# =============================================================================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "paul_graham_essays"

# =============================================================================
# Embedding Model
# =============================================================================
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32

# =============================================================================
# Reranker Model
# =============================================================================
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 3  # Keep top 3 after reranking

# =============================================================================
# LLM Settings
# =============================================================================
LLM_MODEL = "llama-3.1-8b-instant"  # Groq's Llama 3.1 8B (lower token usage)
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024

# =============================================================================
# Chunking Settings
# =============================================================================
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens

# =============================================================================
# Retrieval Settings
# =============================================================================
BASIC_TOP_K = 5  # Basic RAG retrieves top 5
HYBRID_TOP_K = 10  # Hybrid retrieves top 10, then reranks to top 3

# =============================================================================
# Memory Settings
# =============================================================================
# Basic RAG: keeps ALL messages (will explode)
# Optimized RAG: summarizes after this many tokens
SUMMARY_BUFFER_MAX_TOKENS = 1000  # Summarize when history exceeds this

# =============================================================================
# Data Paths
# =============================================================================
DATA_DIR = "data/paul_graham"

# =============================================================================
# Sparse Vector Settings (for hybrid search)
# =============================================================================
SPARSE_MODEL = "Qdrant/bm25"  # BM25 for keyword matching
