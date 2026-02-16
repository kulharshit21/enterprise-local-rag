"""
Generation quality metrics: Exact Match, F1, Faithfulness, Context Utilization.
"""

import re
from typing import List
from collections import Counter


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation, extra spaces."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, reference: str) -> float:
    """
    Exact Match: 1.0 if normalized prediction equals normalized reference, else 0.0.
    """
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def f1_score(prediction: str, reference: str) -> float:
    """
    Token-level F1 score between prediction and reference.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # Count common tokens
    common = sum((pred_counter & ref_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)

    return 2 * (precision * recall) / (precision + recall)


def context_utilization(answer: str, context_chunks: List[str]) -> float:
    """
    Measure how much of the retrieved context was utilized in the answer.

    Computes the fraction of context chunks that have token overlap
    with the generated answer.
    """
    if not context_chunks:
        return 0.0

    answer_tokens = set(normalize_text(answer).split())
    utilized_count = 0

    for chunk in context_chunks:
        chunk_tokens = set(normalize_text(chunk).split())
        overlap = answer_tokens & chunk_tokens
        # Consider chunk utilized if more than 10% of its tokens appear in answer
        if len(overlap) > max(1, len(chunk_tokens) * 0.1):
            utilized_count += 1

    return utilized_count / len(context_chunks)


def answer_relevance(answer: str, query: str) -> float:
    """
    Simple relevance score: token overlap between answer and query.
    Higher overlap suggests the answer addresses the query.
    """
    answer_tokens = set(normalize_text(answer).split())
    query_tokens = set(normalize_text(query).split())

    if not query_tokens:
        return 0.0

    overlap = answer_tokens & query_tokens
    return len(overlap) / len(query_tokens)
