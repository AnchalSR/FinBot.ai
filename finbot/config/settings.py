"""
Configuration settings for FinBot.

This module loads environment variables and provides configuration
for the entire application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application configuration settings."""

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # LLM Configuration
    USE_OPENAI: bool = os.getenv("USE_OPENAI", "false").lower() == "true"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # HuggingFace Configuration
    HF_MODEL_NAME: str = os.getenv(
        "HF_MODEL_NAME",
        "mistralai/Mistral-7B-Instruct-v0.1"
    )
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    
    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "all-MiniLM-L6-v2"
    )
    
    # FAISS Configuration
    FAISS_INDEX_PATH: str = os.getenv(
        "FAISS_INDEX_PATH",
        "embeddings/faiss_index"
    )
    
    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_DOCUMENTS: int = int(os.getenv("TOP_K_DOCUMENTS", "5"))
    
    # Documents Configuration - FIXED: Use consistent naming
    DOCUMENTS_FOLDER: str = os.getenv("DOCUMENTS_FOLDER", "data/documents")
    DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "data/documents")  # Alias for compatibility
    ALLOWED_EXTENSIONS: list = ["pdf", "txt"]
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_PATH: str = os.getenv("LOG_PATH", "logs/finbot.log")
    
    # Model Configuration
    DEVICE: str = os.getenv("DEVICE", "cpu")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Fine-tuning Configuration
    FINETUNE_EPOCHS: int = int(os.getenv("FINETUNE_EPOCHS", "3"))
    FINETUNE_BATCH_SIZE: int = int(os.getenv("FINETUNE_BATCH_SIZE", "8"))
    FINETUNE_LEARNING_RATE: float = float(os.getenv("FINETUNE_LEARNING_RATE", "2e-4"))
    
    # Streamlit Configuration
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_THEME: str = os.getenv("STREAMLIT_THEME", "light")


# Create a global settings instance
settings = Settings()


def print_config():
    """Print active configuration (for debugging)."""
    config_vars = {k: v for k, v in vars(settings).items() if not k.startswith("_")}
    for key, value in sorted(config_vars.items()):
        # Mask sensitive keys
        if "API_KEY" in key or "TOKEN" in key:
            display_value = "***MASKED***"
        else:
            display_value = value
        print(f"{key}: {display_value}")
