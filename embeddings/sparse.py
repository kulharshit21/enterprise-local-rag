"""
BM25 sparse indexing for keyword-based retrieval.
"""

import os
import pickle
from typing import List, Dict, Any, Tuple, Optional
import re

from rank_bm25 import BM25Okapi
from config import settings


class SparseIndexer:
    """
    BM25 sparse index for keyword-based retrieval.
    Complements dense retrieval in the hybrid search pipeline.
    """

    def __init__(self, index_path: str = None):
        self.index_path = index_path or settings.BM25_INDEX_PATH
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization with basic cleaning."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        # Remove very short tokens
        return [t for t in tokens if len(t) > 1]

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Build BM25 index from chunk dictionaries.

        Each chunk dict must have:
        - 'chunk_id': unique identifier
        - 'content': text content
        - 'metadata': dict with role_access, doc_id, etc.
        """
        self.documents = chunks
        self.tokenized_corpus = [self._tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        role_filter: Optional[List[str]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search the BM25 index.

        Returns list of (chunk_dict, score) tuples sorted by relevance.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Pair chunks with scores and filter by role
        results = []
        for i, (doc, score) in enumerate(zip(self.documents, scores)):
            if score <= 0:
                continue

            # Apply role filter
            if role_filter:
                doc_roles = doc.get("metadata", {}).get("role_access", ["viewer"])
                if not any(role in doc_roles for role in role_filter):
                    continue

            results.append((doc, float(score)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self):
        """Serialize index to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        data = {
            "documents": self.documents,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        if not os.path.exists(self.index_path):
            return False

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        return True
