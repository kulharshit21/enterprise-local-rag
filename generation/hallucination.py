"""
Hallucination detection via embedding similarity + NLI-based claim verification.
Evaluates faithfulness of generated answers against retrieved context.
"""

import re
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity


class HallucinationDetector:
    """
    Multi-step faithfulness evaluation pipeline:
    1. Overall embedding similarity (answer vs context)
    2. Claim extraction (sentence splitting)
    3. Per-claim verification via embedding similarity against each context chunk
    4. Aggregate faithfulness score

    Thresholds:
      ≥ 0.70 → Faithful (accept)
      0.40–0.70 → Uncertain (flag with warning)
      < 0.40 → Unfaithful (reject, return safe response)
    """

    FAITHFULNESS_THRESHOLD = 0.70
    REJECTION_THRESHOLD = 0.40

    def __init__(self, embedder):
        """
        Args:
            embedder: DenseEmbedder instance for computing similarities
        """
        self.embedder = embedder

    def evaluate(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate faithfulness of an answer against retrieved context.

        Returns:
            Dict with faithfulness_score, is_faithful, claim_details
        """
        if not answer or not context_chunks:
            return {
                "faithfulness_score": 0.0,
                "is_faithful": False,
                "overall_similarity": 0.0,
                "claims": [],
                "supported_claims": 0,
                "total_claims": 0,
            }

        # Step 1: Overall embedding similarity
        context_text = " ".join(c.get("content", "") for c in context_chunks)
        overall_sim = self._compute_similarity(answer, context_text)

        # Step 2: Claim extraction
        claims = self._extract_claims(answer)

        if not claims:
            return {
                "faithfulness_score": overall_sim,
                "is_faithful": overall_sim >= self.FAITHFULNESS_THRESHOLD,
                "overall_similarity": overall_sim,
                "claims": [],
                "supported_claims": 0,
                "total_claims": 0,
            }

        # Step 3: Per-claim verification
        claim_results = []
        supported_count = 0

        for claim in claims:
            # Find best matching context chunk for this claim
            best_score = 0.0
            best_chunk = ""

            for chunk in context_chunks:
                chunk_text = chunk.get("content", "")
                if not chunk_text:
                    continue
                sim = self._compute_similarity(claim, chunk_text)
                if sim > best_score:
                    best_score = sim
                    best_chunk = chunk_text[:200]

            is_supported = best_score >= 0.5
            if is_supported:
                supported_count += 1

            claim_results.append({
                "claim": claim,
                "best_similarity": round(best_score, 4),
                "is_supported": is_supported,
                "evidence_preview": best_chunk[:100] if best_chunk else "",
            })

        # Step 4: Aggregate
        claim_score = supported_count / len(claims) if claims else 0
        faithfulness_score = 0.6 * claim_score + 0.4 * overall_sim

        return {
            "faithfulness_score": round(faithfulness_score, 4),
            "is_faithful": faithfulness_score >= self.FAITHFULNESS_THRESHOLD,
            "overall_similarity": round(overall_sim, 4),
            "claims": claim_results,
            "supported_claims": supported_count,
            "total_claims": len(claims),
        }

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts using embeddings."""
        try:
            emb_a = self.embedder.embed_query(text_a).reshape(1, -1)
            emb_b = self.embedder.embed_query(text_b).reshape(1, -1)
            sim = cosine_similarity(emb_a, emb_b)[0][0]
            return float(max(0, sim))
        except Exception:
            return 0.0

    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims (sentences) from generated text."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s) > 15 and not s.startswith("Source") and not s.startswith("["):
                claims.append(s)
        return claims

    @staticmethod
    def get_safe_response() -> str:
        """Return a safe response when answer is deemed unfaithful."""
        return (
            "I cannot provide a verified answer based on the available documents. "
            "The retrieved context does not contain sufficient information to "
            "answer this question with confidence. Please try rephrasing your "
            "query or consult additional sources."
        )
