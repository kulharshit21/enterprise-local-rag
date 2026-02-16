"""
Sparse retriever using BM25 keyword search.
"""

from typing import List, Dict, Any, Optional

from embeddings.sparse import SparseIndexer
from config import settings


class SparseRetriever:
    """BM25-based keyword retrieval with role filtering."""

    def __init__(self, indexer: SparseIndexer = None):
        self.indexer = indexer or SparseIndexer()

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        role_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K chunks by BM25 keyword relevance.

        Returns list of dicts with 'chunk_id', 'content', 'metadata', 'score'.
        """
        top_k = top_k or settings.TOP_K

        results = self.indexer.search(
            query=query,
            top_k=top_k,
            role_filter=role_filter,
        )

        # Normalize to same format as dense retriever
        formatted = []
        for chunk_dict, score in results:
            formatted.append({
                "chunk_id": chunk_dict.get("chunk_id", ""),
                "content": chunk_dict.get("content", ""),
                "metadata": chunk_dict.get("metadata", {}),
                "score": score,
            })

        return formatted
