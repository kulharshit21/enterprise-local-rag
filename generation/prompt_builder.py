"""
Prompt engineering for grounded RAG responses.
"""

from typing import List, Dict, Any

import tiktoken

from config import settings


class PromptBuilder:
    """
    Builds structured prompts for grounded question answering.
    Manages token budgets and formats context with citation markers.
    """

    SYSTEM_PROMPT = """You are a grounded enterprise assistant. You MUST follow these rules strictly:

1. ONLY answer using the provided context below. Do NOT use any prior knowledge.
2. If the context does not contain sufficient information to answer the question, say: "Insufficient evidence found in indexed documents."
3. Provide citation markers [1], [2], etc. referencing the context sources.
4. Be concise, precise, and professional.
5. If the question asks about data in a table, reference the table explicitly.
6. Never fabricate information or statistics not present in the context."""

    def __init__(self, max_context_tokens: int = 3000, model: str = "cl100k_base"):
        self.max_context_tokens = max_context_tokens
        self.encoding = tiktoken.get_encoding(model)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def build_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build a complete prompt with system instructions and context.

        Returns:
            Dict with 'system', 'user', 'citations' for LLM input
        """
        # Format context with citation markers
        context_parts = []
        citations = []
        total_tokens = 0

        for i, chunk in enumerate(context_chunks):
            marker = f"[{i+1}]"
            content = chunk.get("content", "")
            source = chunk.get("metadata", {}).get("source_file", "unknown")
            doc_id = chunk.get("metadata", {}).get("doc_id", chunk.get("chunk_id", ""))

            # Format the context entry
            entry = f"{marker} Source: {source}\n{content}"
            entry_tokens = self.count_tokens(entry)

            # Check token budget
            if total_tokens + entry_tokens > self.max_context_tokens:
                break

            context_parts.append(entry)
            citations.append({
                "marker": marker,
                "doc_id": str(doc_id),
                "chunk_id": chunk.get("chunk_id", ""),
                "source_file": str(source),
            })
            total_tokens += entry_tokens

        # Build context block
        context_block = "\n\n---\n\n".join(context_parts)

        # Build user message
        user_message = f"""Context:
{context_block}

---

User Query: {query}

Instructions: Answer the query using ONLY the context above. Include citation markers [1], [2], etc. If evidence is insufficient, say so explicitly."""

        return {
            "system": self.SYSTEM_PROMPT,
            "user": user_message,
            "citations": citations,
            "context_tokens": total_tokens,
        }
