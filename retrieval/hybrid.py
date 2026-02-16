"""
Hybrid retriever combining dense and sparse results via Reciprocal Rank Fusion.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict

from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from config import settings


class HybridRetriever:
    """
    Combines dense (semantic) and sparse (BM25) retrieval using
    Reciprocal Rank Fusion (RRF) for robust hybrid search.

    RRF Score = Σ 1 / (k + rank_i) for each ranking list i
    """

    RRF_K = 60  # Standard RRF constant

    def __init__(
        self,
        dense_retriever: DenseRetriever = None,
        sparse_retriever: SparseRetriever = None,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or SparseRetriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        role_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using hybrid search with RRF merging.

        Args:
            query: User query string
            top_k: Number of results to return
            role_filter: List of roles for access filtering

        Returns:
            Merged and re-ranked list of chunk results
        """
        top_k = top_k or settings.TOP_K

        # Retrieve from both sources (fetch more to allow fusion)
        fetch_k = top_k * 3

        dense_results = self.dense_retriever.retrieve(
            query, top_k=fetch_k, role_filter=role_filter
        )
        sparse_results = self.sparse_retriever.retrieve(
            query, top_k=fetch_k, role_filter=role_filter
        )

        # Apply RRF fusion
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        RRF Score for document d = Σ weight_i / (k + rank_i(d))
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Score dense results
        for rank, result in enumerate(dense_results):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] += self.dense_weight / (self.RRF_K + rank + 1)
            chunk_map[chunk_id] = result

        # Score sparse results
        for rank, result in enumerate(sparse_results):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] += self.sparse_weight / (self.RRF_K + rank + 1)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final results
        fused_results = []
        for chunk_id in sorted_ids:
            result = chunk_map[chunk_id].copy()
            result["rrf_score"] = rrf_scores[chunk_id]
            result["retrieval_method"] = "hybrid_rrf"
            fused_results.append(result)

        return fused_results
