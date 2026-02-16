"""
Dense retriever using FAISS vector store.
Encodes query with BGE prefix and searches FAISS index with role filtering.
"""

from typing import List, Dict, Any, Optional
from embeddings.dense import DenseEmbedder
from embeddings.vector_store import VectorStore


class DenseRetriever:
    """
    Retrieves documents using dense embeddings (BGE-large-en) + FAISS.

    Pipeline:
    1. Encode query with BGE retrieval prefix
    2. Search FAISS index (inner product for L2-normalized vectors = cosine sim)
    3. Apply role-based post-filtering
    """

    def __init__(self, embedder: DenseEmbedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        role_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K documents by dense similarity.

        Args:
            query: Search query string
            top_k: Number of results to return
            role_filter: Allowed roles for RBAC filtering

        Returns:
            List of result dicts with chunk_id, content, metadata, score
        """
        # Encode query with BGE prefix
        query_vector = self.embedder.embed_query(query)

        # Search FAISS index
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            role_filter=role_filter,
        )

        return results
