"""
Unit tests for evaluation metrics.
"""

import pytest
from evaluation.retrieval_metrics import precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k
from evaluation.generation_metrics import exact_match, f1_score, context_utilization, normalize_text


class TestRetrievalMetrics:
    """Tests for retrieval quality metrics."""

    def test_precision_at_k_perfect(self):
        """Test P@K with all relevant results."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_precision_at_k_partial(self):
        """Test P@K with some irrelevant results."""
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=5) == 0.6

    def test_precision_at_k_none(self):
        """Test P@K with no relevant results."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_recall_at_k(self):
        """Test R@K captures fraction of relevant docs."""
        retrieved = ["a", "b", "x"]
        relevant = {"a", "b", "c", "d"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.5

    def test_recall_at_k_full(self):
        """Test R@K with all relevant captured."""
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=2) == 1.0

    def test_mrr_first_position(self):
        """Test MRR when relevant doc is first."""
        assert mean_reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_mrr_second_position(self):
        """Test MRR when relevant doc is second."""
        assert mean_reciprocal_rank(["x", "a"], {"a"}) == 0.5

    def test_mrr_not_found(self):
        """Test MRR when relevant doc is not found."""
        assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_ndcg_at_k(self):
        """Test NDCG computation."""
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        score = ndcg_at_k(retrieved, relevant, k=3)
        assert 0 < score <= 1.0


class TestGenerationMetrics:
    """Tests for generation quality metrics."""

    def test_exact_match_true(self):
        """Test exact match with matching texts."""
        assert exact_match("Hello World", "hello world") == 1.0

    def test_exact_match_false(self):
        """Test exact match with different texts."""
        assert exact_match("Hello", "World") == 0.0

    def test_f1_score_perfect(self):
        """Test F1 with identical texts."""
        score = f1_score("the cat sat on the mat", "the cat sat on the mat")
        assert score == 1.0

    def test_f1_score_partial(self):
        """Test F1 with partial overlap."""
        score = f1_score("the cat", "the cat sat on the mat")
        assert 0 < score < 1.0

    def test_f1_score_no_overlap(self):
        """Test F1 with no common tokens."""
        score = f1_score("hello", "world goodbye")
        assert score == 0.0

    def test_context_utilization(self):
        """Test context utilization calculation."""
        answer = "The cat sat on the mat in the house"
        contexts = [
            "The cat sat on the mat",
            "Dogs play in the park",
            "The house is very big and beautiful",
        ]
        util = context_utilization(answer, contexts)
        assert 0 < util <= 1.0

    def test_normalize_text(self):
        """Test text normalization."""
        assert normalize_text("Hello, World!") == "hello world"
        assert normalize_text("  spaces  ") == "spaces"
