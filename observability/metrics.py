"""
Prometheus metrics for production monitoring.
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# --- Counters ---
QUERY_COUNTER = Counter(
    "rag_queries_total",
    "Total number of RAG queries processed",
    ["user_role", "status"],
)

HALLUCINATION_COUNTER = Counter(
    "rag_hallucinations_total",
    "Total number of hallucination detections",
    ["severity"],
)

AUTH_COUNTER = Counter(
    "rag_auth_attempts_total",
    "Total authentication attempts",
    ["status"],
)

# --- Histograms ---
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "Query processing latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "LLM generation latency in seconds",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

FAITHFULNESS_SCORE = Histogram(
    "rag_faithfulness_score",
    "Distribution of faithfulness scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

TOKEN_USAGE = Histogram(
    "rag_token_usage",
    "Token usage per query",
    ["token_type"],
    buckets=[50, 100, 250, 500, 1000, 2000, 4000],
)

# --- Gauges ---
ACTIVE_QUERIES = Gauge(
    "rag_active_queries",
    "Number of currently processing queries",
)

INDEX_SIZE = Gauge(
    "rag_index_size",
    "Number of chunks in the index",
    ["collection"],
)

# --- Info ---
SYSTEM_INFO = Info(
    "rag_system",
    "RAG system configuration info",
)


def record_query_metrics(
    user_role: str,
    status: str,
    total_latency_s: float,
    retrieval_latency_s: float,
    generation_latency_s: float,
    faithfulness: float,
    prompt_tokens: int,
    completion_tokens: int,
):
    """Record all metrics for a completed query."""
    QUERY_COUNTER.labels(user_role=user_role, status=status).inc()
    QUERY_LATENCY.observe(total_latency_s)
    RETRIEVAL_LATENCY.observe(retrieval_latency_s)
    GENERATION_LATENCY.observe(generation_latency_s)
    FAITHFULNESS_SCORE.observe(faithfulness)
    TOKEN_USAGE.labels(token_type="prompt").observe(prompt_tokens)
    TOKEN_USAGE.labels(token_type="completion").observe(completion_tokens)

    if faithfulness < 0.5:
        HALLUCINATION_COUNTER.labels(severity="high").inc()
    elif faithfulness < 0.7:
        HALLUCINATION_COUNTER.labels(severity="medium").inc()
