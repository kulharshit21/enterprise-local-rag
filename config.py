"""
Centralized configuration for the Enterprise RAG System.
Reads from .env file and environment variables.
All local — no paid API keys required.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable binding."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # ── LLM (Local LLaMA) ──
    LLAMA_MODEL_PATH: str = "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    LLM_CONTEXT_LENGTH: int = 4096
    LLM_MAX_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.1
    LLM_GPU_LAYERS: int = -1  # -1 = all layers to GPU

    # ── Embeddings ──
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_DEVICE: str = "cuda"

    # ── Reranker ──
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"

    # ── Retrieval ──
    TOP_K: int = 5
    RERANK_TOP_K: int = 20
    DENSE_WEIGHT: float = 0.6
    SPARSE_WEIGHT: float = 0.4
    RRF_K: int = 60

    # ── FAISS ──
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    FAISS_USE_GPU: bool = False  # Default false for CI; true in production

    # ── BM25 ──
    BM25_INDEX_PATH: str = "./data/bm25_index.pkl"

    # ── Chunking ──
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Security ──
    JWT_SECRET: str = "dev-secret-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    RATE_LIMIT_REQUESTS: int = 30
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ── Observability ──
    LOG_FORMAT: str = "text"  # "json" for structured logging
    LOG_LEVEL: str = "INFO"

    # ── Data Paths ──
    DOCUMENTS_DIR: str = "./data/documents"
    IMAGES_DIR: str = "./data/images"
    METADATA_DB_PATH: str = "./data/metadata.db"
    USERS_DB_PATH: str = "./data/users_db.json"


# Singleton
settings = Settings()


def ensure_directories():
    """Create all required data directories."""
    dirs = [
        settings.DOCUMENTS_DIR,
        settings.IMAGES_DIR,
        os.path.dirname(settings.FAISS_INDEX_PATH) or ".",
        os.path.dirname(settings.BM25_INDEX_PATH) or ".",
        os.path.dirname(settings.METADATA_DB_PATH) or ".",
        os.path.dirname(settings.USERS_DB_PATH) or ".",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
