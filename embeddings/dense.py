"""
Dense Embeddings via BGE-large-en-v1.5 (1024-dim).
GPU-accelerated, float16, with BGE query-prefix protocol.
"""

import numpy as np
from typing import List, Optional
from config import settings


class DenseEmbedder:
    """
    Wraps BAAI/bge-large-en-v1.5 for dense text embeddings.
    - 1024-dimensional output
    - Float16 inference for VRAM efficiency
    - BGE query-prefix for improved retrieval quality
    - L2-normalized for cosine similarity via inner product
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        self._model = None

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = 512

            # Float16 for GPU VRAM efficiency (~40% savings, <0.1% quality loss)
            if self.device == "cuda":
                self._model.half()
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a query with BGE retrieval prefix.
        Returns L2-normalized 1024-dim vector.
        """
        prefixed = f"{self.QUERY_PREFIX}{query}"
        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embedding, dtype=np.float32)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode document chunks (no prefix needed for BGE passages).
        Returns (N, 1024) normalized float32 array.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return settings.EMBEDDING_DIMENSION
