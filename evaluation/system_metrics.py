"""
System-level performance metrics: latency, token usage, cost estimation.
"""

import time
from typing import Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str = ""
    total_latency_ms: float = 0
    retrieval_latency_ms: float = 0
    generation_latency_ms: float = 0
    reranking_latency_ms: float = 0
    hallucination_check_ms: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0
    num_chunks_retrieved: int = 0


class SystemMetricsTracker:
    """
    Tracks and aggregates system performance metrics across queries.
    """

    # Cost per 1M tokens (approximate for gpt-4o-mini)
    INPUT_COST_PER_1M = 0.15   # $0.15 per 1M input tokens
    OUTPUT_COST_PER_1M = 0.60  # $0.60 per 1M output tokens

    def __init__(self):
        self.queries: List[QueryMetrics] = []

    def record_query(self, metrics: QueryMetrics):
        """Record metrics for a completed query."""
        # Estimate cost
        metrics.estimated_cost_usd = (
            (metrics.prompt_tokens / 1_000_000) * self.INPUT_COST_PER_1M
            + (metrics.completion_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        )
        self.queries.append(metrics)

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        if not self.queries:
            return {"total_queries": 0}

        latencies = [q.total_latency_ms for q in self.queries]
        tokens = [q.total_tokens for q in self.queries]
        costs = [q.estimated_cost_usd for q in self.queries]

        return {
            "total_queries": len(self.queries),
            "latency": {
                "mean_ms": round(sum(latencies) / len(latencies), 2),
                "min_ms": round(min(latencies), 2),
                "max_ms": round(max(latencies), 2),
                "p50_ms": round(sorted(latencies)[len(latencies) // 2], 2),
                "p95_ms": round(
                    sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0], 2
                ),
            },
            "tokens": {
                "total": sum(tokens),
                "mean_per_query": round(sum(tokens) / len(tokens), 2),
            },
            "cost": {
                "total_usd": round(sum(costs), 6),
                "mean_per_query_usd": round(sum(costs) / len(costs), 6),
            },
        }


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.start_time = 0
        self.elapsed_ms = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
