"""
Intelligent document chunking with semantic awareness and token limits.
Uses tiktoken for accurate token counting.
"""

import re
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass, field

import tiktoken


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticChunker:
    """
    Splits documents into semantically coherent chunks respecting token limits.

    Strategy:
    1. Split by natural boundaries (headings, paragraphs, tables)
    2. Merge small segments up to CHUNK_SIZE tokens
    3. Add overlap for context continuity
    """

    def __init__(
        self, chunk_size: int = 512, chunk_overlap: int = 50, model: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(model)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))

    def chunk_document(
        self, content: str, doc_id: str, metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
        """
        Split document content into chunks.

        Respects:
        - Heading boundaries (# ## ###)
        - Paragraph boundaries
        - Table boundaries ([TABLE]...[/TABLE])
        - Token size limits
        - Overlap for continuity
        """
        metadata = metadata or {}
        segments = self._split_into_segments(content)

        # Merge small segments and split large ones
        merged = self._merge_segments(segments)

        chunks = []
        for i, segment_text in enumerate(merged):
            chunk = Chunk(
                content=segment_text.strip(),
                doc_id=doc_id,
                chunk_index=i,
                token_count=self.count_tokens(segment_text),
                metadata={
                    **metadata,
                    "section_type": self._detect_section_type(segment_text),
                },
            )
            chunks.append(chunk)

        # Add overlap
        chunks = self._add_overlap(chunks)

        return chunks

    def _split_into_segments(self, text: str) -> List[str]:
        """Split text by semantic boundaries."""
        # Split by headings, double newlines, and table markers
        pattern = r"((?:^|\n)#{1,6}\s.+|\n\n+|\[TABLE\]|\[/TABLE\])"
        parts = re.split(pattern, text)

        segments = []
        current = ""
        for part in parts:
            if not part.strip():
                continue
            # Check if this is a boundary marker
            if re.match(r"^#{1,6}\s", part.strip()) or part.strip() in (
                "[TABLE]",
                "[/TABLE]",
            ):
                if current.strip():
                    segments.append(current.strip())
                current = part
            else:
                current += part

        if current.strip():
            segments.append(current.strip())

        return segments

    def _merge_segments(self, segments: List[str]) -> List[str]:
        """Merge small segments and split large ones to respect chunk_size."""
        merged = []
        current = ""
        current_tokens = 0

        for segment in segments:
            seg_tokens = self.count_tokens(segment)

            # If single segment exceeds chunk_size, force-split it
            if seg_tokens > self.chunk_size:
                if current.strip():
                    merged.append(current.strip())
                    current = ""
                    current_tokens = 0
                # Force split by sentences
                sub_chunks = self._force_split(segment)
                merged.extend(sub_chunks)
                continue

            # If adding this segment exceeds limit, flush current
            if current_tokens + seg_tokens > self.chunk_size:
                if current.strip():
                    merged.append(current.strip())
                current = segment
                current_tokens = seg_tokens
            else:
                current += "\n\n" + segment if current else segment
                current_tokens += seg_tokens

        if current.strip():
            merged.append(current.strip())

        return merged

    def _force_split(self, text: str) -> List[str]:
        """Force-split a large segment by sentences, with hard token fallback."""
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # If sentence splitting didn't help (no sentence boundaries), do hard token split
        if len(sentences) == 1 and self.count_tokens(text) > self.chunk_size:
            return self._hard_token_split(text)

        chunks = []
        current = ""
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            # If a single sentence exceeds chunk_size, hard-split it
            if sent_tokens > self.chunk_size:
                if current.strip():
                    chunks.append(current.strip())
                    current = ""
                    current_tokens = 0
                chunks.extend(self._hard_token_split(sentence))
                continue

            if current_tokens + sent_tokens > self.chunk_size:
                if current.strip():
                    chunks.append(current.strip())
                current = sentence
                current_tokens = sent_tokens
            else:
                current += " " + sentence if current else sentence
                current_tokens += sent_tokens

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _hard_token_split(self, text: str) -> List[str]:
        """Split text by hard token boundaries when no sentence boundaries exist."""
        tokens = self.encoding.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
        return chunks

    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """Add overlap text from previous chunk to maintain context."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].content
            prev_tokens = self.encoding.encode(prev_text)

            # Take last `chunk_overlap` tokens from previous chunk
            overlap_tokens = prev_tokens[-self.chunk_overlap :]
            overlap_text = self.encoding.decode(overlap_tokens)

            chunks[i].content = f"...{overlap_text}\n\n{chunks[i].content}"
            chunks[i].token_count = self.count_tokens(chunks[i].content)

        return chunks

    @staticmethod
    def _detect_section_type(text: str) -> str:
        """Detect the type of content in a segment."""
        if "[TABLE]" in text:
            return "table"
        if re.match(r"^#{1,6}\s", text.strip()):
            return "heading"
        if text.strip().startswith(("- ", "* ", "1.", "•")):
            return "list"
        return "paragraph"
