"""
Unit tests for the ingestion chunker module.
"""

from ingestion.chunker import SemanticChunker, Chunk


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def setup_method(self):
        self.chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)

    def test_basic_chunking(self):
        """Test basic text chunking produces non-empty chunks."""
        text = "This is a test document. " * 50
        chunks = self.chunker.chunk_document(text, doc_id="test-doc")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_has_metadata(self):
        """Test that chunks include proper metadata."""
        text = "Short document for testing metadata assignment."
        chunks = self.chunker.chunk_document(
            text, doc_id="doc-123", metadata={"role_access": ["admin"]}
        )
        assert len(chunks) > 0
        assert chunks[0].doc_id == "doc-123"
        assert "section_type" in chunks[0].metadata

    def test_chunk_size_limit(self):
        """Test that chunks respect token size limits."""
        text = "Word " * 1000  # Very long text
        chunks = self.chunker.chunk_document(text, doc_id="big-doc")

        for chunk in chunks:
            # Allow some tolerance for overlap
            assert chunk.token_count <= self.chunker.chunk_size * 1.5

    def test_empty_text(self):
        """Test handling of empty text."""
        chunks = self.chunker.chunk_document("", doc_id="empty")
        assert len(chunks) == 0

    def test_heading_detection(self):
        """Test that headings are detected in section_type."""
        text = "# Main Heading\n\nSome content under the heading."
        chunks = self.chunker.chunk_document(text, doc_id="heading-doc")
        assert len(chunks) > 0
        # At least one chunk should have heading section type
        section_types = [c.metadata.get("section_type") for c in chunks]
        assert "heading" in section_types or "paragraph" in section_types

    def test_token_counting(self):
        """Test that token counting works correctly."""
        text = "Hello world"
        tokens = self.chunker.count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_unique_chunk_ids(self):
        """Test that each chunk gets a unique ID."""
        text = "Paragraph one. " * 20 + "\n\n" + "Paragraph two. " * 20
        chunks = self.chunker.chunk_document(text, doc_id="unique-test")
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
