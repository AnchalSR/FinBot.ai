"""
Text preprocessing utilities for document processing.

This module provides functions for cleaning, tokenizing, and
chunking documents for the RAG pipeline.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import PyPDF2

logger = logging.getLogger(__name__)


def load_pdf_documents(pdf_path: str) -> str:
    """
    Load text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
        
    Raises:
        FileNotFoundError: If PDF file not found
        ValueError: If PDF cannot be read
    """
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
                
        logger.info(f"Successfully loaded PDF: {pdf_path} ({len(text)} characters)")
        return text
        
    except Exception as e:
        logger.error(f"Error loading PDF {pdf_path}: {str(e)}")
        raise ValueError(f"Cannot read PDF: {str(e)}")


def load_text_documents(txt_path: str) -> str:
    """
    Load text from a TXT file.
    
    Args:
        txt_path: Path to the TXT file
        
    Returns:
        Content of the text file
        
    Raises:
        FileNotFoundError: If file not found
    """
    try:
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Text file not found: {txt_path}")
            
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()
            
        logger.info(f"Successfully loaded TXT: {txt_path} ({len(text)} characters)")
        return text
        
    except Exception as e:
        logger.error(f"Error loading TXT {txt_path}: {str(e)}")
        raise


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Removes extra whitespace, special characters, and normalizes formatting.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.!?,;:\-\']', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk (characters)
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        logger.warning("Empty text provided for chunking")
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        
    logger.info(f"Created {len(chunks)} chunks from text (size={chunk_size}, overlap={overlap})")
    return chunks


def process_documents_folder(
    folder_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str, str]]:
    """
    Process all documents in a folder.
    
    Args:
        folder_path: Path to folder containing documents
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of processed documents with metadata
        
    Example:
        >>> docs = process_documents_folder("data/documents", chunk_size=500)
        >>> print(f"Processed {len(docs)} document chunks")
    """
    documents = []
    folder = Path(folder_path)
    
    if not folder.exists():
        logger.warning(f"Documents folder not found: {folder_path}")
        return documents
    
    # Process PDF files
    for pdf_file in folder.glob("*.pdf"):
        try:
            text = load_pdf_documents(str(pdf_file))
            text = clean_text(text)
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    "content": chunk,
                    "source": pdf_file.name,
                    "chunk_id": i,
                    "type": "pdf"
                })
                
            logger.info(f"Processed PDF: {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}")
    
    # Process TXT files
    for txt_file in folder.glob("*.txt"):
        try:
            text = load_text_documents(str(txt_file))
            text = clean_text(text)
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    "content": chunk,
                    "source": txt_file.name,
                    "chunk_id": i,
                    "type": "txt"
                })
                
            logger.info(f"Processed TXT: {txt_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {txt_file.name}: {str(e)}")
    
    logger.info(f"Total documents processed: {len(documents)}")
    return documents


def tokenize_text(text: str, model_name: str = "bert-base-uncased") -> List[str]:
    """
    Tokenize text using a specified model.
    
    Args:
        text: Text to tokenize
        model_name: Name of the tokenizer model
        
    Returns:
        List of tokens
    """
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.tokenize(text)
        
        return tokens
        
    except Exception as e:
        logger.error(f"Error tokenizing text: {str(e)}")
        # Fallback to simple split
        return text.split()


def get_document_stats(documents: List[Dict[str, str]]) -> Dict:
    """
    Calculate statistics about processed documents.
    
    Args:
        documents: List of document chunks
        
    Returns:
        Dictionary with statistics
    """
    if not documents:
        return {
            "total_chunks": 0,
            "total_characters": 0,
            "average_chunk_size": 0,
            "sources": []
        }
    
    sources = {}
    total_chars = 0
    
    for doc in documents:
        source = doc.get("source", "unknown")
        chars = len(doc.get("content", ""))
        total_chars += chars
        
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
    
    return {
        "total_chunks": len(documents),
        "total_characters": total_chars,
        "average_chunk_size": total_chars // len(documents) if documents else 0,
        "sources": sources
    }
