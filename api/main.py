"""
FastAPI application entry point.
Configures CORS, mounts routes, and sets up middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from api.routes import (
    auth_router,
    query_router,
    ingest_router,
    eval_router,
    observability_router,
    health_router,
)
from observability.logger import setup_logging
from observability import metrics as obs_metrics
from config import settings, ensure_directories


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Enterprise RAG System",
        description=(
            "Production-grade Retrieval-Augmented Generation API with "
            "hybrid search, RBAC, hallucination detection, and observability."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Prometheus metrics endpoint ---
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # --- Include routers ---
    app.include_router(auth_router)
    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(eval_router)
    app.include_router(observability_router)
    app.include_router(health_router)

    # --- Startup event ---
    @app.on_event("startup")
    async def startup():
        setup_logging()
        ensure_directories()

        # Set system info metric
        obs_metrics.SYSTEM_INFO.info(
            {
                "llm_model": settings.LLAMA_MODEL_PATH,
                "embedding_model": settings.EMBEDDING_MODEL,
                "chunk_size": str(settings.CHUNK_SIZE),
                "top_k": str(settings.TOP_K),
            }
        )

    # --- Root redirect ---
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": "Enterprise RAG System API",
            "docs": "/docs",
            "dashboard": "/dashboard",
            "health": "/health",
        }

    return app


# Application instance
app = create_app()
