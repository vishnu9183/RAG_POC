"""
Document loading and chunking utilities.
"""
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
import config


def load_documents(data_dir: str) -> list:
    """
    Load all .txt files from the given directory.
    Returns a list of LangChain Document objects.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Run 'python scripts/download_essays.py' first."
        )

    documents = []
    txt_files = sorted(data_path.glob("*.txt"))

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        title = file_path.stem.replace("_", " ").title()
        documents.append(Document(
            page_content=text,
            metadata={
                "title": title,
                "filename": file_path.name,
                "source": str(file_path)
            }
        ))

    print(f"Loaded {len(documents)} documents from {data_dir}.")
    return documents


def chunk_documents(documents: list) -> list:
    """
    Split documents into token-based chunks.
    """
    splitter = TokenTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents.")
    return chunks
