"""
Retrieval Augmented Generation (RAG) pipeline for FinBot.

This module implements the core RAG functionality including:
- Document embedding
- Vector database management
- Similarity search
- Context retrieval
"""

import os
import logging
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from finbot.config.settings import settings
from finbot.utils.preprocess import process_documents_folder, get_document_stats

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Retrieval Augmented Generation Pipeline.
    
    Manages document embeddings, vector database, and retrieval operations.
    """
    
    def __init__(
        self,
        embedding_model: str = settings.EMBEDDING_MODEL,
        faiss_index_path: str = settings.FAISS_INDEX_PATH,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        device: str = settings.DEVICE
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Name of the embedding model to use
            faiss_index_path: Path to store/load FAISS index
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            device: Device to run embeddings on (cpu/cuda)
        """
        self.embedding_model_name = embedding_model
        self.model_name = embedding_model  # Alias for compatibility
        self.faiss_index_path = faiss_index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"✅ Embedding model loaded successfully. Dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Failed to load embedding model '{embedding_model}': {str(e)}")
        
        # Initialize FAISS index
        self.faiss_index: Optional[faiss.IndexFlatL2] = None
        self.documents: List[Dict[str, str]] = []
        self.metadata: List[Dict] = []
        
        # Load existing index if available
        self._load_index()
        
        logger.info("✅ RAG Pipeline initialized successfully")
    
    @property
    def index_size(self) -> int:
        """Get current index size (number of documents)."""
        return len(self.documents)
    
    def _load_index(self) -> None:
        """Load FAISS index and documents from disk."""
        try:
            index_file = f"{self.faiss_index_path}.index"
            metadata_file = f"{self.faiss_index_path}_metadata.pkl"
            docs_file = f"{self.faiss_index_path}_documents.pkl"
            
            if os.path.exists(index_file) and os.path.exists(docs_file):
                self.faiss_index = faiss.read_index(index_file)
                
                with open(docs_file, "rb") as f:
                    self.documents = pickle.load(f)
                
                with open(metadata_file, "rb") as f:
                    self.metadata = pickle.load(f)
                
                logger.info(
                    f"Loaded FAISS index with {len(self.documents)} documents"
                )
            else:
                logger.info("No existing FAISS index found. Creating new one.")
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
    
    def _save_index(self) -> None:
        """Save FAISS index and documents to disk."""
        try:
            # Create embeddings directory if it doesn't exist
            Path(self.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
            
            index_file = f"{self.faiss_index_path}.index"
            metadata_file = f"{self.faiss_index_path}_metadata.pkl"
            docs_file = f"{self.faiss_index_path}_documents.pkl"
            
            # Save FAISS index
            faiss.write_index(self.faiss_index, index_file)
            
            # Save documents and metadata
            with open(docs_file, "wb") as f:
                pickle.dump(self.documents, f)
            
            with open(metadata_file, "wb") as f:
                pickle.dump(self.metadata, f)
            
            logger.info(
                f"Saved FAISS index with {len(self.documents)} documents"
            )
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Add documents to the FAISS index.
        
        Args:
            documents: List of documents with 'content' key
        """
        try:
            if not documents:
                logger.warning("No documents provided")
                return
            
            # Extract content for embedding
            texts = [doc.get("content", "") for doc in documents]
            
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=True
            )
            
            # Add to FAISS index
            embeddings = embeddings.astype(np.float32)
            self.faiss_index.add(embeddings)
            
            # Store documents and metadata
            self.documents.extend(documents)
            
            for i, doc in enumerate(documents):
                self.metadata.append({
                    "source": doc.get("source", "unknown"),
                    "chunk_id": doc.get("chunk_id", 0),
                    "type": doc.get("type", "unknown"),
                    "index": len(self.documents) - len(documents) + i
                })
            
            # Save index
            self._save_index()
            
            logger.info(
                f"Added {len(documents)} documents to index. "
                f"Total documents: {len(self.documents)}"
            )
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def retrieve_documents(
        self,
        query: str,
        top_k: int = settings.TOP_K_DOCUMENTS
    ) -> List[Dict[str, str]]:
        """
        Retrieve most relevant documents for a query.
        
        Args:
            query: User query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        try:
            if len(self.documents) == 0:
                logger.warning("No documents in index")
                return []
            
            # Encode query
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True
            ).astype(np.float32)
            
            # Search in FAISS
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Retrieve documents
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc["score"] = float(distance)  # Lower distance = higher relevance
                    doc["similarity"] = 1 / (1 + float(distance))
                    results.append(doc)
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def build_context(
        self,
        query: str,
        top_k: int = settings.TOP_K_DOCUMENTS
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Build context string from retrieved documents.
        
        Args:
            query: User query
            top_k: Number of documents to use for context
            
        Returns:
            Tuple of (context_string, retrieved_documents)
        """
        try:
            # Retrieve relevant documents
            documents = self.retrieve_documents(query, top_k)
            
            if not documents:
                logger.warning("No documents retrieved for context")
                return "", []
            
            # Build context string
            context_parts = []
            sources = set()
            
            for i, doc in enumerate(documents, 1):
                content = doc.get("content", "")
                source = doc.get("source", "Unknown")
                similarity = doc.get("similarity", 0)
                
                context_parts.append(
                    f"[Document {i} - {source} (Relevance: {similarity:.2%})]\n{content}\n"
                )
                sources.add(source)
            
            context = "\n".join(context_parts)
            
            logger.info(
                f"Built context with {len(documents)} documents "
                f"from {len(sources)} sources"
            )
            
            return context, documents
            
        except Exception as e:
            logger.error(f"Error building context: {str(e)}")
            return "", []
    
    def load_documents_from_folder(self, folder_path: str) -> None:
        """
        Load all documents from a folder and add to index.
        
        Args:
            folder_path: Path to folder containing documents
        """
        try:
            logger.info(f"Loading documents from {folder_path}")
            
            documents = process_documents_folder(
                folder_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            if documents:
                self.add_documents(documents)
                stats = get_document_stats(documents)
                logger.info(f"Document stats: {stats}")
            else:
                logger.warning(f"No documents found in {folder_path}")
                
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
    
    def get_stats(self) -> Dict:
        """Get statistics about the RAG pipeline."""
        return {
            "total_documents": len(self.documents),
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dim,
            "index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "device": self.device,
            "sources": self._get_sources_count()
        }
    
    def _get_sources_count(self) -> Dict[str, int]:
        """Get count of documents per source."""
        sources = {}
        for doc in self.documents:
            source = doc.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        return sources
    
    def clear_index(self) -> None:
        """Clear all documents and reset index."""
        try:
            self.documents = []
            self.metadata = []
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self._save_index()
            logger.info("Cleared FAISS index")
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}")
            raise


def create_rag_pipeline(
    embedding_model: str = settings.EMBEDDING_MODEL,
    device: str = settings.DEVICE
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.
    
    Args:
        embedding_model: Name of embedding model
        device: Device to use (cpu/cuda)
        
    Returns:
        Initialized RAGPipeline instance
    """
    return RAGPipeline(embedding_model=embedding_model, device=device)
