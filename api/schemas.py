"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


# ------------------------------------------------------------------ #
#  AUTH
# ------------------------------------------------------------------ #

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: str = ""
    role: str = Field(default="viewer", pattern="^(admin|researcher|analyst|viewer)$")


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


# ------------------------------------------------------------------ #
#  QUERY
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class CitationResponse(BaseModel):
    doc_id: str = ""
    chunk_id: str = ""
    source_file: str = ""


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]
    confidence_score: float
    metadata: Dict[str, Any] = {}


# ------------------------------------------------------------------ #
#  INGESTION
# ------------------------------------------------------------------ #

class IngestRequest(BaseModel):
    directory: Optional[str] = None
    file_path: Optional[str] = None
    role_access: List[str] = Field(default=["viewer"])
    sensitivity: str = Field(default="internal", pattern="^(public|internal|confidential|restricted)$")


class IngestResponse(BaseModel):
    message: str
    documents_loaded: int = 0
    chunks_created: int = 0
    index_size: int = 0


# ------------------------------------------------------------------ #
#  EVALUATION
# ------------------------------------------------------------------ #

class EvaluationResponse(BaseModel):
    aggregate_metrics: Dict[str, Any]
    dataset_size: int
    timestamp: str
    detailed_results: List[Dict[str, Any]] = []


# ------------------------------------------------------------------ #
#  HEALTH
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    index_size: int = 0
    timestamp: str = ""
