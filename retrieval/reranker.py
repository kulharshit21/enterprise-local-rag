"""
Cross-encoder reranker using BGE-reranker-large.
Reranks top-K candidates from hybrid retrieval for precision improvement.
"""

from typing import List, Dict, Any
from config import settings


class CrossEncoderReranker:
    """
    Wraps BAAI/bge-reranker-large for passage reranking.

    Takes ⟨query, passage⟩ pairs and scores them with a cross-encoder.
    Significantly more accurate than bi-encoder similarity but
    O(n) in candidate count, so only applied to top-20 candidates.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._reranker = None

    @property
    def reranker(self):
        """Lazy-load the reranker model."""
        if self._reranker is None:
            from FlagEmbedding import FlagReranker

            self._reranker = FlagReranker(
                self.model_name,
                use_fp16=True,  # Float16 for VRAM efficiency
            )
        return self._reranker

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate passages using cross-encoder scores.

        Args:
            query: The search query
            candidates: List of retrieval results with 'content' field
            top_k: Number of top results to return after reranking

        Returns:
            Reranked list of candidates (highest relevance first)
        """
        top_k = top_k or settings.TOP_K

        if not candidates:
            return []

        # Build ⟨query, passage⟩ pairs
        pairs = [[query, c.get("content", "")] for c in candidates]

        # Score all pairs
        scores = self.reranker.compute_score(pairs)

        # Handle single result (returns float instead of list)
        if isinstance(scores, (int, float)):
            scores = [scores]

        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(
            candidates, key=lambda x: x.get("rerank_score", 0), reverse=True
        )

        return reranked[:top_k]
