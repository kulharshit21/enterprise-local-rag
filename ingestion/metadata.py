"""
Metadata extraction and tagging for document chunks.
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from ingestion.loader import Document
from ingestion.chunker import Chunk, SemanticChunker
from config import settings


class MetadataExtractor:
    """
    Processes documents through chunking and enriches each chunk
    with comprehensive metadata for retrieval filtering.
    """

    def __init__(self, chunker: SemanticChunker = None):
        self.chunker = chunker or SemanticChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    def process_document(
        self,
        document: Document,
        role_access: Optional[List[str]] = None,
        sensitivity: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Chunk a document and enrich each chunk with metadata.

        Returns list of Chunks ready for embedding and indexing.
        """
        # Override metadata if provided
        if role_access:
            document.metadata["role_access"] = role_access
        if sensitivity:
            document.metadata["sensitivity"] = sensitivity

        # Ensure role_access exists
        if "role_access" not in document.metadata:
            document.metadata["role_access"] = ["viewer"]

        # Chunk the document
        chunks = self.chunker.chunk_document(
            content=document.content,
            doc_id=document.doc_id,
            metadata=document.metadata,
        )

        # Enrich metadata on each chunk
        for chunk in chunks:
            chunk.metadata.update({
                "doc_id": document.doc_id,
                "source_file": document.source_file,
                "doc_type": document.doc_type,
                "role_access": document.metadata.get("role_access", ["viewer"]),
                "sensitivity": document.metadata.get("sensitivity", "internal"),
                "ingested_at": datetime.utcnow().isoformat(),
            })

        return chunks

    def process_documents(
        self,
        documents: List[Document],
        role_access: Optional[List[str]] = None,
        sensitivity: Optional[str] = None,
    ) -> List[Chunk]:
        """Process multiple documents, returning all chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self.process_document(doc, role_access, sensitivity)
            all_chunks.extend(chunks)
        return all_chunks
