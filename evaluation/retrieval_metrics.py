"""
Retrieval quality metrics: Precision@K, Recall@K, MRR.
"""

from typing import List, Set


def precision_at_k(
    retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5
) -> float:
    """
    Precision@K: fraction of retrieved documents in top-K that are relevant.

    P@K = |relevant ∩ retrieved[:K]| / K
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """
    Recall@K: fraction of relevant documents captured in top-K.

    R@K = |relevant ∩ retrieved[:K]| / |relevant|
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Mean Reciprocal Rank: 1/position of first relevant document.

    MRR = 1 / rank of first relevant result (0 if none found)
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Uses binary relevance (1 if relevant, 0 otherwise).
    """
    import math

    top_k = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # +2 because rank is 1-indexed

    # Ideal DCG (all relevant documents at the top)
    ideal_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))

    if idcg == 0:
        return 0.0

    return dcg / idcg
