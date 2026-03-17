"""
Ingest documents into Qdrant.

This script:
1. Loads all text files from data/paul_graham/
2. Chunks them into ~500 token pieces
3. Creates embeddings using BGE
4. Stores in Qdrant with both dense and sparse vectors

Run this once after downloading essays:
    python ingest.py
"""
import sys
from tqdm import tqdm
from qdrant_client.http import models

import config
from shared.loader import load_documents, chunk_documents
from shared.embeddings import get_embedding_model, embed_texts
from shared.vectorstore import get_qdrant_client, create_collection


def ingest_documents(recreate_collection: bool = False):
    """
    Main ingestion pipeline.
    """
    print("=" * 60)
    print("RAG Document Ingestion")
    print("=" * 60)
    
    # Step 1: Load documents
    print("\n[1/4] Loading documents...")
    try:
        documents = load_documents(config.DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python scripts/download_essays.py' first to download essays.")
        sys.exit(1)
    
    if not documents:
        print("No documents found. Run download script first.")
        sys.exit(1)
    
    # Step 2: Chunk documents
    print("\n[2/4] Chunking documents...")
    chunks = chunk_documents(documents)
    
    # Step 3: Create collection
    print("\n[3/4] Setting up Qdrant collection...")
    create_collection(recreate=recreate_collection)
    
    # Step 4: Embed and upload
    print("\n[4/4] Embedding and uploading to Qdrant...")
    
    client = get_qdrant_client()
    embedding_model = get_embedding_model()
    
    # Process in batches
    batch_size = config.EMBEDDING_BATCH_SIZE
    points = []
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        
        # Get texts for embedding
        texts = [chunk.page_content for chunk in batch]
        
        # Create embeddings
        embeddings = embedding_model.embed_documents(texts)
        
        # Create points for Qdrant
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point_id = i + j
            
            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense": embedding
                },
                payload={
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
            )
            points.append(point)
        
        # Upload batch
        if len(points) >= 100:
            client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=points
            )
            points = []
    
    # Upload remaining
    if points:
        client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=points
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    
    info = client.get_collection(config.COLLECTION_NAME)
    print(f"Collection: {config.COLLECTION_NAME}")
    print(f"Total points: {info.points_count}")
    print(f"Vector dimension: {config.EMBEDDING_DIMENSION}")
    print(f"\nYou can now run: python app.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument(
        "--recreate", 
        action="store_true",
        help="Delete and recreate the collection"
    )
    
    args = parser.parse_args()
    ingest_documents(recreate_collection=args.recreate)
