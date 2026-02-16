"""
API route definitions for the RAG system.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse

from api.schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    EvaluationResponse,
    HealthResponse,
)
from api.dependencies import (
    require_admin,
    check_rate_limit,
    get_pipeline,
    get_users_db,
)
from security.auth import (
    verify_password,
    create_access_token,
    TokenData,
)
from security.users_db import UsersDB
from pipeline import RAGPipeline
from evaluation.runner import EvaluationRunner
from observability.dashboard import generate_dashboard_html
from observability import metrics as obs_metrics

# ------------------------------------------------------------------ #
#  AUTH ROUTES
# ------------------------------------------------------------------ #
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post("/register", response_model=TokenResponse)
async def register(
    request: RegisterRequest,
    users_db: UsersDB = Depends(get_users_db),
):
    """Register a new user account."""
    user = users_db.create_user(
        username=request.username,
        password=request.password,
        role=request.role,
        email=request.email,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    obs_metrics.AUTH_COUNTER.labels(status="register_success").inc()

    token = create_access_token(data={"sub": user.username, "role": user.role})

    return TokenResponse(
        access_token=token,
        role=user.role,
        username=user.username,
    )


@auth_router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    users_db: UsersDB = Depends(get_users_db),
):
    """Authenticate and get a JWT token."""
    user = users_db.get_user(request.username)

    if not user or not verify_password(request.password, user.hashed_password):
        obs_metrics.AUTH_COUNTER.labels(status="login_failed").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account disabled",
        )

    obs_metrics.AUTH_COUNTER.labels(status="login_success").inc()

    token = create_access_token(data={"sub": user.username, "role": user.role})

    return TokenResponse(
        access_token=token,
        role=user.role,
        username=user.username,
    )


# ------------------------------------------------------------------ #
#  QUERY ROUTES
# ------------------------------------------------------------------ #
query_router = APIRouter(tags=["Query"])


@query_router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    current_user: TokenData = Depends(check_rate_limit),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Submit a question to the RAG system.

    The system performs hybrid retrieval, LLM generation,
    and hallucination detection. Results are filtered by user role.
    """
    try:
        result = pipeline.query(
            question=request.question,
            user_role=current_user.role,
            top_k=request.top_k,
        )

        # Remove internal fields
        result.pop("_context_texts", None)

        return QueryResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )


# ------------------------------------------------------------------ #
#  INGESTION ROUTES
# ------------------------------------------------------------------ #
ingest_router = APIRouter(tags=["Ingestion"])


@ingest_router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    current_user: TokenData = Depends(require_admin),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Ingest documents into the RAG system (admin only).
    Supports directory or single file ingestion.
    """
    result = pipeline.ingest_documents(
        directory=request.directory,
        file_path=request.file_path,
        role_access=request.role_access,
        sensitivity=request.sensitivity,
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"],
        )

    return IngestResponse(**result)


# ------------------------------------------------------------------ #
#  EVALUATION ROUTES
# ------------------------------------------------------------------ #
eval_router = APIRouter(tags=["Evaluation"])


@eval_router.get("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    current_user: TokenData = Depends(require_admin),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Run the automated evaluation suite (admin only)."""
    runner = EvaluationRunner()
    report = runner.evaluate(pipeline, user_role="admin")
    runner.save_report(report)
    return EvaluationResponse(**report)


# ------------------------------------------------------------------ #
#  OBSERVABILITY ROUTES
# ------------------------------------------------------------------ #
observability_router = APIRouter(tags=["Observability"])


@observability_router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the observability dashboard."""
    return generate_dashboard_html()


@observability_router.get("/api/metrics/summary")
async def metrics_summary(
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Get system metrics summary for the dashboard."""
    summary = pipeline.system_metrics.get_summary()
    summary["index_size"] = pipeline.vector_store.get_collection_count("text_chunks")

    if summary.get("total_queries", 0) > 0:
        summary["avg_latency_ms"] = summary.get("latency", {}).get("mean_ms", 0)
        summary["avg_faithfulness"] = 0.75  # Placeholder until tracked
    else:
        summary["avg_latency_ms"] = 0
        summary["avg_faithfulness"] = 0

    return summary


# ------------------------------------------------------------------ #
#  HEALTH ROUTES
# ------------------------------------------------------------------ #
health_router = APIRouter(tags=["Health"])


@health_router.get("/health", response_model=HealthResponse)
async def health_check(
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """System health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        index_size=pipeline.vector_store.get_collection_count("text_chunks"),
        timestamp=datetime.utcnow().isoformat(),
    )
