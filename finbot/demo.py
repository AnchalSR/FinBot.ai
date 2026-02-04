#!/usr/bin/env python
"""
FinBot Demo Script

This script demonstrates how to use FinBot programmatically.
"""

import logging
import os
from pathlib import Path

from finbot.config.settings import settings
from finbot.backend.rag import create_rag_pipeline
from finbot.utils.preprocess import process_documents_folder, get_document_stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_rag():
    """Demo 1: Basic RAG pipeline"""
    print("\n" + "="*60)
    print("Demo 1: Basic RAG Pipeline")
    print("="*60)
    
    # Create RAG pipeline
    rag = create_rag_pipeline(
        embedding_model=settings.EMBEDDING_MODEL,
        device=settings.DEVICE
    )
    
    print(f"✓ RAG pipeline created")
    print(f"  - Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"  - Device: {settings.DEVICE}")
    
    # Create sample documents for demo
    sample_docs = [
        {
            "content": "Compound interest is the process of earning interest on interest. "
                      "It's calculated using the formula A = P(1 + r/n)^(nt).",
            "source": "finance_basics.txt",
            "chunk_id": 0,
            "type": "txt"
        },
        {
            "content": "Diversification is an investment strategy to spread risk across "
                      "different asset classes and sectors.",
            "source": "investment_guide.txt",
            "chunk_id": 1,
            "type": "txt"
        },
        {
            "content": "Index funds track a market index and provide passive investment "
                      "with lower fees than active management.",
            "source": "investment_guide.txt",
            "chunk_id": 2,
            "type": "txt"
        }
    ]
    
    # Add documents
    rag.add_documents(sample_docs)
    print(f"✓ Added {len(sample_docs)} sample documents")
    
    # Test retrieval
    query = "What is compound interest?"
    results = rag.retrieve_documents(query, top_k=2)
    
    print(f"\n✓ Retrieved {len(results)} documents for query: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"  [{i}] {result['source']} - Similarity: {result['similarity']:.2%}")
    
    # Get statistics
    stats = rag.get_stats()
    print(f"\n✓ Pipeline Statistics:")
    print(f"  - Total Documents: {stats['total_documents']}")
    print(f"  - Embedding Dimension: {stats['embedding_dimension']}")


def demo_document_loading():
    """Demo 2: Load documents from folder"""
    print("\n" + "="*60)
    print("Demo 2: Document Loading")
    print("="*60)
    
    # Check if documents folder exists
    doc_folder = Path(settings.DOCUMENTS_PATH)
    
    if not doc_folder.exists():
        print(f"ℹ Documents folder not found: {settings.DOCUMENTS_PATH}")
        print("To use this feature, add PDF/TXT files to data/documents/")
        return
    
    # List files
    pdf_files = list(doc_folder.glob("*.pdf"))
    txt_files = list(doc_folder.glob("*.txt"))
    
    print(f"✓ Found {len(pdf_files)} PDF files and {len(txt_files)} TXT files")
    
    if pdf_files or txt_files:
        # Create RAG pipeline
        rag = create_rag_pipeline()
        
        # Load documents
        documents = process_documents_folder(
            settings.DOCUMENTS_PATH,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        if documents:
            rag.add_documents(documents)
            stats = get_document_stats(documents)
            
            print(f"\n✓ Loaded and processed documents:")
            print(f"  - Total Chunks: {stats['total_chunks']}")
            print(f"  - Total Characters: {stats['total_characters']:,}")
            print(f"  - Average Chunk Size: {stats['average_chunk_size']}")
            print(f"  - Sources: {stats['sources']}")


def demo_context_building():
    """Demo 3: Build context from query"""
    print("\n" + "="*60)
    print("Demo 3: Context Building")
    print("="*60)
    
    # Create RAG pipeline with sample data
    rag = create_rag_pipeline()
    
    sample_docs = [
        {
            "content": "Asset allocation is the strategy of dividing your portfolio "
                      "among different asset classes like stocks, bonds, and cash.",
            "source": "portfolio.txt",
            "chunk_id": 0,
            "type": "txt"
        },
        {
            "content": "Risk management involves identifying potential losses and taking "
                      "steps to minimize their impact on your investment.",
            "source": "risk.txt",
            "chunk_id": 0,
            "type": "txt"
        }
    ]
    
    rag.add_documents(sample_docs)
    
    # Build context
    query = "How should I manage my investment portfolio?"
    context, docs = rag.build_context(query, top_k=2)
    
    print(f"✓ Query: '{query}'")
    print(f"✓ Retrieved {len(docs)} documents for context")
    print(f"\nContext (first 200 chars):")
    print(f"  {context[:200]}...")
    
    # Show sources
    print(f"\nSources:")
    for doc in docs:
        print(f"  - {doc['source']} (Similarity: {doc['similarity']:.2%})")


def demo_api_usage():
    """Demo 4: API usage example"""
    print("\n" + "="*60)
    print("Demo 4: API Usage (Requires running backend)")
    print("="*60)
    
    import requests
    import json
    
    api_url = f"http://{settings.API_HOST}:{settings.API_PORT}"
    
    print(f"ℹ API URL: {api_url}")
    print("\nExample API calls:")
    
    # Health check
    print(f"\n1. Health Check:")
    print(f"   curl {api_url}/health")
    
    # Chat
    print(f"\n2. Chat Endpoint:")
    print(f"   curl -X POST {api_url}/chat \\")
    print(f"     -H 'Content-Type: application/json' \\")
    query_data = json.dumps({"query": "What is diversification?"})
    print(f"     -d '{query_data}'")
    
    # Upload
    print(f"\n3. Upload Documents:")
    print(f"   curl -X POST {api_url}/upload \\")
    print(f"     -F 'files=@finance_guide.pdf'")
    
    # Status
    print(f"\n4. Get Status:")
    print(f"   curl {api_url}/status")
    
    # Try to connect
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        if response.status_code == 200:
            print(f"\n✓ API is running!")
            print(f"  Status: {response.json().get('status')}")
        else:
            print(f"\n✗ API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot connect to API at {api_url}")
        print(f"  Make sure backend is running: python -m backend.api")


def main():
    """Run all demos"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  FinBot - Demonstration Script".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    # Run demos
    demo_basic_rag()
    demo_document_loading()
    demo_context_building()
    demo_api_usage()
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run backend: python -m backend.api")
    print("2. Run frontend: streamlit run frontend/app.py")
    print("3. Add documents to data/documents/")
    print("4. Visit http://localhost:8501")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
