"""
FinBot - Financial Advisor Chatbot
A production-ready RAG-powered financial advisor chatbot with FastAPI backend.

This module provides the REST API endpoints for the FinBot chatbot system.
It handles document uploads, RAG queries, and system status.
"""

__version__ = "1.0.0"
__author__ = "Anchal Kumar"
__description__ = "Financial Advisor Chatbot with RAG"

import logging
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Absolute imports using finbot package
from finbot.config.settings import settings
from finbot.backend.rag import RAGPipeline, create_rag_pipeline
from finbot.utils.preprocess import process_documents_folder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Global State ====================

# RAG pipeline - initialized on startup
rag_pipeline: Optional[RAGPipeline] = None

# ==================== Pydantic Models ====================

class ChatRequest(BaseModel):
    """Chat request model."""
    query: str = Field(..., description="User's question")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    include_sources: bool = Field(default=True, description="Include source documents in response")


class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str = Field(..., description="AI's answer")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents")
    query: str = Field(..., description="Original query")
    timestamp: str = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="API status")
    timestamp: str = Field(..., description="Health check timestamp")
    documents_count: int = Field(default=0, description="Number of indexed documents")
    embedding_model: str = Field(default="", description="Embedding model name")
    rag_initialized: bool = Field(default=False, description="RAG pipeline initialized")


class StatusResponse(BaseModel):
    """System status response."""
    status: str = Field(..., description="Overall status")
    rag_initialized: bool = Field(..., description="RAG pipeline ready")
    documents_count: int = Field(..., description="Total indexed documents")
    embedding_dimension: int = Field(..., description="Embedding vector dimension")
    device: str = Field(..., description="Compute device (cpu/cuda)")
    embedding_model: str = Field(..., description="Active embedding model")
    timestamp: str = Field(..., description="Status timestamp")


class UploadResponse(BaseModel):
    """Document upload response."""
    status: str = Field(..., description="Upload status")
    uploaded_files: List[str] = Field(..., description="List of uploaded files")
    message: str = Field(..., description="Status message")


class DocumentsResponse(BaseModel):
    """Documents list response."""
    total_documents: int = Field(..., description="Total indexed documents")
    sources: Dict[str, int] = Field(..., description="Source to chunk count mapping")


class DeleteResponse(BaseModel):
    """Document deletion response."""
    status: str = Field(..., description="Deletion status")
    removed_documents: int = Field(..., description="Number of removed documents")
    remaining_documents: int = Field(..., description="Remaining documents after deletion")


# ==================== Startup/Shutdown ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan - startup and shutdown events.
    
    This is where we initialize heavy resources like the RAG pipeline.
    """
    # Startup
    logger.info("🚀 FinBot Backend Starting...")
    logger.info(f"Settings - Device: {settings.DEVICE}, Embedding: {settings.EMBEDDING_MODEL}")
    logger.info(f"Documents Folder: {settings.DOCUMENTS_FOLDER}")
    
    try:
        global rag_pipeline
        
        # Initialize RAG pipeline
        logger.info("📚 Initializing RAG pipeline...")
        try:
            rag_pipeline = create_rag_pipeline(
                embedding_model=settings.EMBEDDING_MODEL,
                device=settings.DEVICE
            )
            logger.info("✅ RAG pipeline created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create RAG pipeline: {str(e)}", exc_info=True)
            raise
        
        # Load documents from folder if it exists
        docs_folder = Path(settings.DOCUMENTS_FOLDER)
        logger.info(f"Checking documents folder: {docs_folder.absolute()}")
        
        if docs_folder.exists() and docs_folder.is_dir():
            logger.info(f"📖 Loading documents from {docs_folder}...")
            try:
                rag_pipeline.load_documents_from_folder(str(docs_folder))
                logger.info(f"✅ RAG pipeline ready with {rag_pipeline.index_size} documents indexed")
                if rag_pipeline.index_size == 0:
                    logger.warning("⚠️  No documents found in folder, but RAG is operational")
            except Exception as e:
                logger.error(f"❌ Error loading documents: {str(e)}", exc_info=True)
                logger.info("   RAG pipeline ready but empty. Upload documents via /upload endpoint")
        else:
            logger.info(f"⚠️  Documents folder not found at {docs_folder.absolute()}")
            logger.info("   Creating folder: mkdir -p " + str(docs_folder))
            docs_folder.mkdir(parents=True, exist_ok=True)
            logger.info("   You can upload documents via the /upload endpoint")
        
        logger.info("✨ Backend initialized successfully!")
        
    except Exception as e:
        logger.critical(f"❌ CRITICAL ERROR during startup: {str(e)}", exc_info=True)
        logger.critical("Application will continue but RAG will not function. Check logs above.")
        rag_pipeline = None
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("🛑 FinBot Backend Shutting Down...")
    logger.info("✅ Cleanup complete")


# ==================== FastAPI Application ====================

app = FastAPI(
    title="FinBot - Financial Advisor Chatbot",
    description="A production-ready RAG-powered financial advisor chatbot API",
    version=__version__,
    lifespan=lifespan
)

# ==================== CORS Middleware ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Root Endpoint ====================

@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint - verify API is running.
    
    Returns:
        Welcome message with API info
    """
    return {
        "message": "🤖 FinBot API is running!",
        "version": __version__,
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health"
    }


# ==================== Health Check ====================

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with current status
    """
    global rag_pipeline
    
    docs_count = 0
    model_name = "all-MiniLM-L6-v2"
    
    if rag_pipeline is not None:
        docs_count = rag_pipeline.index_size
        model_name = rag_pipeline.model_name if hasattr(rag_pipeline, 'model_name') else model_name
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        documents_count=docs_count,
        embedding_model=model_name,
        rag_initialized=rag_pipeline is not None
    )


# ==================== Status Endpoint ====================

@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """
    Get detailed system status.
    
    Returns:
        StatusResponse with system information
    """
    global rag_pipeline
    
    if rag_pipeline is None:
        return StatusResponse(
            status="initializing",
            rag_initialized=False,
            documents_count=0,
            embedding_dimension=384,
            device=settings.DEVICE,
            embedding_model="all-MiniLM-L6-v2",
            timestamp=datetime.now().isoformat()
        )
    
    return StatusResponse(
        status="ready",
        rag_initialized=True,
        documents_count=rag_pipeline.index_size,
        embedding_dimension=rag_pipeline.embedding_dim,
        device=settings.DEVICE,
        embedding_model=rag_pipeline.model_name if hasattr(rag_pipeline, 'model_name') else "all-MiniLM-L6-v2",
        timestamp=datetime.now().isoformat()
    )


# ==================== Chat Endpoint ====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint - answer user questions with RAG context.
    
    Args:
        request: ChatRequest with query and parameters
        
    Returns:
        ChatResponse with answer and sources
        
    Raises:
        HTTPException: If RAG pipeline not initialized or error occurs
    """
    global rag_pipeline
    
    logger.info(f"Chat request received. RAG initialized: {rag_pipeline is not None}")
    
    if rag_pipeline is None:
        logger.error("❌ RAG pipeline is None - not initialized properly")
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Check server logs for initialization errors."
        )
    
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Retrieve relevant documents
        context, sources = rag_pipeline.build_context(
            query=request.query,
            top_k=request.top_k
        )
        
        # Build context string
        if context.strip():
            context_str = f"Based on provided documents:\n{context}\n\nQuestion: {request.query}"
        else:
            context_str = request.query
        
        # Generate answer using LLM
        try:
            if settings.USE_OPENAI:
                # OpenAI integration
                from openai import OpenAI
                client = OpenAI(api_key=settings.OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful financial advisor. Provide accurate, helpful responses."},
                        {"role": "user", "content": context_str}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                answer = response.choices[0].message.content
            else:
                # HuggingFace integration
                from transformers import pipeline
                try:
                    llm = pipeline(
                        "text-generation",
                        model="mistralai/Mistral-7B-Instruct-v0.1",
                        device=0 if settings.DEVICE == "cuda" else -1
                    )
                    response = llm(context_str, max_length=200, do_sample=True)
                    answer = response[0]["generated_text"]
                except Exception:
                    # Fallback: return context-based answer
                    answer = context if context else "I don't have enough information to answer this question."
        
        except ImportError:
            # Fallback if no LLM available
            answer = context if context else "I don't have enough information to answer this question."
        
        # Format sources
        formatted_sources = []
        if request.include_sources and sources:
            for source in sources:
                formatted_sources.append({
                    "source": source.get("source", "unknown"),
                    "similarity": source.get("similarity", 0.0),
                    "chunk_id": source.get("chunk_id", 0)
                })
        
        logger.info(f"✅ Generated response with {len(formatted_sources)} sources")
        
        return ChatResponse(
            answer=answer,
            sources=formatted_sources,
            query=request.query,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# ==================== Document Management ====================

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> UploadResponse:
    """
    Upload documents for RAG indexing.
    
    Args:
        files: List of PDF or TXT files to upload
        background_tasks: FastAPI background tasks
        
    Returns:
        UploadResponse with status
        
    Raises:
        HTTPException: If upload fails
    """
    global rag_pipeline
    
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized"
        )
    
    try:
        uploaded_files = []
        docs_folder = Path(settings.DOCUMENTS_FOLDER)
        docs_folder.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        for file in files:
            if file.filename:
                file_path = docs_folder / file.filename
                content = await file.read()
                with open(file_path, 'wb') as f:
                    f.write(content)
                uploaded_files.append(file.filename)
                logger.info(f"✅ Saved: {file.filename}")
        
        # Process documents in background
        if uploaded_files:
            background_tasks.add_task(
                _process_uploaded_documents,
                str(docs_folder)
            )
        
        return UploadResponse(
            status="success",
            uploaded_files=uploaded_files,
            message=f"Uploaded {len(uploaded_files)} file(s). Processing in background."
        )
    
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


def _process_uploaded_documents(folder_path: str) -> None:
    """
    Process uploaded documents and update RAG index.
    Runs in background.
    
    Args:
        folder_path: Path to documents folder
    """
    global rag_pipeline
    
    try:
        logger.info(f"🔄 Processing documents from {folder_path}...")
        if rag_pipeline:
            rag_pipeline.load_documents_from_folder(folder_path)
            logger.info(f"✅ Documents processed. Index size: {rag_pipeline.index_size}")
    except Exception as e:
        logger.error(f"Error processing documents: {e}")


@app.get("/documents", response_model=DocumentsResponse)
async def get_documents() -> DocumentsResponse:
    """
    Get list of indexed documents.
    
    Returns:
        DocumentsResponse with document information
        
    Raises:
        HTTPException: If RAG pipeline not initialized
    """
    global rag_pipeline
    
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized"
        )
    
    try:
        stats = rag_pipeline.get_stats()
        return DocumentsResponse(
            total_documents=stats.get("total_documents", 0),
            sources=stats.get("sources", {})
        )
    except Exception as e:
        logger.error(f"❌ Error getting documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{source}")
async def delete_document(source: str) -> DeleteResponse:
    """
    Delete a document by source name.
    
    Args:
        source: Document source/filename to delete
        
    Returns:
        DeleteResponse with deletion status
        
    Raises:
        HTTPException: If deletion fails
    """
    global rag_pipeline
    
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized"
        )
    
    try:
        # Remove file from disk
        docs_folder = Path(settings.DOCUMENTS_FOLDER)
        file_path = docs_folder / source
        
        removed_count = 0
        if file_path.exists():
            file_path.unlink()
            logger.info(f"✅ Deleted: {source}")
            removed_count = 1
        
        # Rebuild index without this source
        logger.info("🔄 Rebuilding index...")
        rag_pipeline.load_documents_from_folder(str(docs_folder))
        
        remaining = rag_pipeline.index_size
        
        return DeleteResponse(
            status="success",
            removed_documents=removed_count,
            remaining_documents=remaining
        )
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    logger.info(f"Starting FinBot API v{__version__}...")
    logger.info(f"API Host: {settings.API_HOST}")
    logger.info(f"API Port: {settings.API_PORT}")
    logger.info(f"LLM: {'OpenAI' if settings.USE_OPENAI else 'HuggingFace'}")
    
    uvicorn.run(
        "finbot.backend.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
