# RAG Comparison POC

A side-by-side comparison of Basic RAG vs Optimized RAG pipelines, demonstrating why production RAG systems need hybrid search, reranking, and conversation summarization.

## What You'll Learn

| Problem | Basic RAG | Optimized RAG |
|---------|-----------|---------------|
| Keyword misses | Semantic-only search | Hybrid (semantic + BM25) |
| Irrelevant chunks | Top-K, no filtering | Cross-encoder reranking |
| Token explosion | Full conversation history | Rolling summarization |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Gradio UI                            │
│  ┌─────────────────────┐     ┌─────────────────────────┐   │
│  │     Basic RAG       │     │    Optimized RAG        │   │
│  │  - Top-5 retrieval  │     │  - Hybrid search        │   │
│  │  - Full history     │     │  - Reranker             │   │
│  │  - No reranking     │     │  - Summary buffer       │   │
│  └─────────────────────┘     └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Qdrant (Docker)                          │
│            Dense vectors + Sparse vectors (BM25)            │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack (All Free)

- **Vector DB**: Qdrant (Docker, local)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (HuggingFace, local)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)
- **LLM**: Groq API (free tier, Llama 3.1 70B)
- **Framework**: LangChain
- **UI**: Gradio
- **Data**: Paul Graham Essays (public domain)

---

## Setup Instructions

### Step 1: Clone and Install Dependencies

```bash
# Create project directory
mkdir rag-comparison-poc && cd rag-comparison-poc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Get API Keys

#### Groq API Key (Free)
1. Go to https://console.groq.com/
2. Sign up (free, no credit card needed)
3. Go to API Keys → Create API Key
4. Copy the key

#### Brave Search API Key (Free - for Phase 2)
1. Go to https://brave.com/search/api/
2. Sign up for free tier (2,000 queries/month)
3. Get your API key

### Step 3: Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your keys
nano .env  # or use any editor
```

Your `.env` should look like:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
BRAVE_API_KEY=BSAxxxxxxxxxxxxxxxxxxxx  # Optional, for Phase 2
```

### Step 4: Start Qdrant with Docker

```bash
# Pull and run Qdrant
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Verify it's running
curl http://localhost:6333/health
# Should return: {"status":"ok"}
```

### Step 5: Download Sample Data

```bash
# Download Paul Graham essays
python scripts/download_essays.py

# This creates data/paul_graham/ with ~200 essays
```

### Step 6: Ingest Documents

```bash
# Load essays into Qdrant
python ingest.py

# This will:
# 1. Load all .txt files from data/paul_graham/
# 2. Chunk them (500 tokens, 50 overlap)
# 3. Create embeddings with BGE
# 4. Store in Qdrant (both dense and sparse vectors)
```

### Step 7: Run the Comparison UI

```bash
# Start the Gradio app
python app.py

# Open http://localhost:7860 in your browser
```

---

## Project Structure

```
rag-comparison-poc/
├── .env.example              # Template for API keys
├── .env                      # Your actual API keys (gitignored)
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration and settings
│
├── shared/
│   ├── __init__.py
│   ├── embeddings.py         # BGE embedding model
│   ├── vectorstore.py        # Qdrant client setup
│   ├── llm.py                # Groq LLM setup
│   └── loader.py             # Document loading + chunking
│
├── basic/
│   ├── __init__.py
│   ├── retriever.py          # Simple top-k retrieval
│   ├── memory.py             # ConversationBufferMemory
│   └── chain.py              # Basic RAG chain
│
├── optimized/
│   ├── __init__.py
│   ├── retriever.py          # Hybrid search + reranker
│   ├── memory.py             # ConversationSummaryBufferMemory
│   └── chain.py              # Optimized RAG chain
│
├── scripts/
│   └── download_essays.py    # Download Paul Graham essays
│
├── data/
│   └── paul_graham/          # Essays stored here
│
├── ingest.py                 # Load docs → Qdrant
├── app.py                    # Gradio comparison UI
└── README.md                 # This file
```

---

## Usage

### Basic Comparison

1. Open http://localhost:7860
2. Type a question about Paul Graham's essays
3. Watch both pipelines respond side-by-side
4. Compare:
   - Token usage (shown below each response)
   - Retrieved chunks (expandable)
   - Response quality

### Test Queries to Try

```
# Semantic search works well for these:
"What does Paul Graham say about startups?"
"How should I approach hiring?"

# Keyword search needed for these (optimized wins):
"What essay mentions Lisp?"
"Find the essay about Y Combinator funding"

# Multi-turn to see token explosion (basic fails):
"Tell me about startups"
"What about funding?"
"How do I find co-founders?"
"What mistakes should I avoid?"
"Tell me more about the first point"
... keep going for 10+ turns
```

---

## Phase 2: MCP Integration (Optional)

To add live web search via Brave:

```bash
# Make sure BRAVE_API_KEY is set in .env
python app.py --enable-mcp

# Now you can ask questions that need current info:
"What are the latest trends in AI startups?"
```

---

## Troubleshooting

### Qdrant won't start
```bash
# Check if port is in use
lsof -i :6333

# Remove old container and restart
docker rm -f qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Out of memory during embedding
```bash
# Use smaller batch size in config.py
EMBEDDING_BATCH_SIZE = 8  # Default is 32
```

### Groq rate limit
```bash
# Free tier: 30 requests/minute
# Add delay between requests or upgrade tier
```

---

## How It Works

### Basic RAG Flow
```
Query → Embed → Top-5 Vector Search → All chunks to prompt
                                            ↓
                              Full conversation history
                                            ↓
                                    LLM generates response
```

### Optimized RAG Flow
```
Query → Embed → Hybrid Search (dense + sparse)
                        ↓
              Cross-encoder reranker (scores each chunk)
                        ↓
              Top-3 most relevant chunks
                        ↓
              Summarized conversation history (~200 tokens)
                        ↓
              LLM generates response
```


