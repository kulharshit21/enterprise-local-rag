"""
Unit tests for retrieval module — RRF merging and role filtering.
"""

import pytest
from retrieval.hybrid import HybridRetriever
from security.rbac import RBACFilter


class TestRRFMerging:
    """Tests for Reciprocal Rank Fusion."""

    def test_rrf_merges_two_lists(self):
        """Test that RRF produces a merged ranked list."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.dense_weight = 0.6
        retriever.sparse_weight = 0.4
        retriever.RRF_K = 60

        dense = [
            {"chunk_id": "a", "content": "doc a", "metadata": {}, "score": 0.9},
            {"chunk_id": "b", "content": "doc b", "metadata": {}, "score": 0.8},
            {"chunk_id": "c", "content": "doc c", "metadata": {}, "score": 0.7},
        ]
        sparse = [
            {"chunk_id": "b", "content": "doc b", "metadata": {}, "score": 5.0},
            {"chunk_id": "d", "content": "doc d", "metadata": {}, "score": 4.0},
            {"chunk_id": "a", "content": "doc a", "metadata": {}, "score": 3.0},
        ]

        fused = retriever._reciprocal_rank_fusion(dense, sparse)

        assert len(fused) == 4  # a, b, c, d
        # b should be top since it appears in both lists at good positions
        chunk_ids = [r["chunk_id"] for r in fused]
        assert "a" in chunk_ids
        assert "b" in chunk_ids
        assert all("rrf_score" in r for r in fused)

    def test_rrf_scores_are_positive(self):
        """Test that RRF scores are positive."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.dense_weight = 0.6
        retriever.sparse_weight = 0.4
        retriever.RRF_K = 60

        dense = [{"chunk_id": "x", "content": "x", "metadata": {}, "score": 0.5}]
        sparse = [{"chunk_id": "y", "content": "y", "metadata": {}, "score": 1.0}]

        fused = retriever._reciprocal_rank_fusion(dense, sparse)
        assert all(r["rrf_score"] > 0 for r in fused)

    def test_rrf_empty_inputs(self):
        """Test RRF with empty inputs."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.dense_weight = 0.6
        retriever.sparse_weight = 0.4
        retriever.RRF_K = 60

        fused = retriever._reciprocal_rank_fusion([], [])
        assert fused == []


class TestRBACFilter:
    """Tests for Role-Based Access Control."""

    def test_admin_sees_all(self):
        """Admin should access all role levels."""
        roles = RBACFilter.get_accessible_roles("admin")
        assert "admin" in roles
        assert "researcher" in roles
        assert "analyst" in roles
        assert "viewer" in roles

    def test_viewer_sees_only_viewer(self):
        """Viewer should only access viewer-level docs."""
        roles = RBACFilter.get_accessible_roles("viewer")
        assert roles == ["viewer"]

    def test_researcher_hierarchy(self):
        """Researcher should access researcher, analyst, and viewer."""
        roles = RBACFilter.get_accessible_roles("researcher")
        assert "researcher" in roles
        assert "analyst" in roles
        assert "viewer" in roles
        assert "admin" not in roles

    def test_can_access_positive(self):
        """Test positive access check."""
        assert RBACFilter.can_access("admin", ["researcher"])
        assert RBACFilter.can_access("researcher", ["viewer", "analyst"])

    def test_can_access_negative(self):
        """Test negative access check (viewer can't access admin docs)."""
        assert not RBACFilter.can_access("viewer", ["admin"])
        assert not RBACFilter.can_access("analyst", ["admin", "researcher"])

    def test_filter_results(self):
        """Test post-retrieval role filtering."""
        results = [
            {"chunk_id": "1", "metadata": {"role_access": "admin"}},
            {"chunk_id": "2", "metadata": {"role_access": "viewer"}},
            {"chunk_id": "3", "metadata": {"role_access": "researcher,admin"}},
        ]

        filtered = RBACFilter.filter_results(results, "viewer")
        assert len(filtered) == 1
        assert filtered[0]["chunk_id"] == "2"

        filtered_admin = RBACFilter.filter_results(results, "admin")
        assert len(filtered_admin) == 3
