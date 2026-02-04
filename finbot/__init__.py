"""
FinBot - Financial Advisor Chatbot

A production-ready RAG-powered financial advisor chatbot.
"""

__version__ = "1.0.0"
__author__ = "Anchal Kumar"
__description__ = "Financial Advisor Chatbot with RAG"

# Correct absolute imports
from finbot.config.settings import settings
from finbot.backend.rag import RAGPipeline, create_rag_pipeline
from finbot.utils.preprocess import process_documents_folder


__all__ = [
    "settings",
    "RAGPipeline",
    "create_rag_pipeline",
    "process_documents_folder"
]
