"""
Structured logging with structlog.
JSON-formatted log entries for production observability.
"""

import uuid
import structlog
from config import settings


def setup_logging():
    """Configure structured logging."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.LOG_FORMAT == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog, settings.LOG_LEVEL, 20)  # Default INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "rag"):
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def generate_query_id() -> str:
    """Generate a unique query ID for request tracing."""
    return str(uuid.uuid4())[:8]


class QueryLogger:
    """
    Structured query logging for the RAG pipeline.

    Logs each query with:
    - query_id
    - retrieval_ids
    - latency_ms
    - hallucination_score
    - final_answer_length
    """

    def __init__(self):
        setup_logging()
        self.logger = get_logger("rag.query")

    def log_query_start(self, query_id: str, query: str, user_role: str):
        """Log the start of a query."""
        self.logger.info(
            "query_started",
            query_id=query_id,
            query=query[:100],  # Truncate for safety
            user_role=user_role,
        )

    def log_retrieval(self, query_id: str, retrieval_ids: list, latency_ms: float):
        """Log retrieval results."""
        self.logger.info(
            "retrieval_completed",
            query_id=query_id,
            retrieval_ids=retrieval_ids[:10],  # Limit logged IDs
            num_results=len(retrieval_ids),
            latency_ms=round(latency_ms, 2),
        )

    def log_generation(self, query_id: str, latency_ms: float, tokens: int):
        """Log generation results."""
        self.logger.info(
            "generation_completed",
            query_id=query_id,
            latency_ms=round(latency_ms, 2),
            total_tokens=tokens,
        )

    def log_hallucination_check(self, query_id: str, score: float, is_faithful: bool):
        """Log hallucination check results."""
        self.logger.info(
            "hallucination_check",
            query_id=query_id,
            hallucination_score=round(score, 4),
            is_faithful=is_faithful,
        )

    def log_query_complete(
        self, query_id: str, total_latency_ms: float, answer_length: int
    ):
        """Log query completion."""
        self.logger.info(
            "query_completed",
            query_id=query_id,
            total_latency_ms=round(total_latency_ms, 2),
            final_answer_length=answer_length,
        )

    def log_error(self, query_id: str, error: str):
        """Log query errors."""
        self.logger.error(
            "query_error",
            query_id=query_id,
            error=error,
        )
