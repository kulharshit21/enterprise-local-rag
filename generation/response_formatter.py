"""
Response formatting with citation extraction.
"""

import re
from typing import List, Dict, Any


class ResponseFormatter:
    """
    Formats LLM responses with structured citations
    and confidence information.
    """

    def format_response(
        self,
        raw_answer: str,
        citations: List[Dict[str, Any]],
        confidence_score: float,
        usage: Dict[str, Any] = None,
        latency_ms: float = 0,
    ) -> Dict[str, Any]:
        """
        Format the final response with citations.

        Returns structured response dict matching the API schema.
        """
        # Extract which citations were actually used in the answer
        used_citations = self._extract_used_citations(raw_answer, citations)

        return {
            "answer": raw_answer,
            "citations": used_citations,
            "confidence_score": round(confidence_score, 4),
            "metadata": {
                "token_usage": usage or {},
                "latency_ms": round(latency_ms, 2),
                "num_sources": len(used_citations),
            },
        }

    @staticmethod
    def _extract_used_citations(
        answer: str, all_citations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract only citations referenced in the answer text."""
        used = []
        # Find all [N] markers in the answer
        markers = re.findall(r"\[(\d+)\]", answer)
        referenced_indices = set(int(m) for m in markers)

        for citation in all_citations:
            marker_num = int(citation["marker"].strip("[]"))
            if marker_num in referenced_indices:
                used.append({
                    "doc_id": citation.get("doc_id", ""),
                    "chunk_id": citation.get("chunk_id", ""),
                    "source_file": citation.get("source_file", ""),
                })

        # If no markers found, include all citations (LLM didn't use markers)
        if not used and all_citations:
            used = [
                {
                    "doc_id": c.get("doc_id", ""),
                    "chunk_id": c.get("chunk_id", ""),
                    "source_file": c.get("source_file", ""),
                }
                for c in all_citations
            ]

        return used
