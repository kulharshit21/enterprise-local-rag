"""
FastAPI dependency injection for auth, pipeline, and services.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from security.auth import decode_access_token, TokenData
from security.rate_limiter import RateLimiter
from pipeline import RAGPipeline
from security.users_db import UsersDB

# Singletons
_pipeline: Optional[RAGPipeline] = None
_users_db: Optional[UsersDB] = None
_rate_limiter = RateLimiter()
_bearer_scheme = HTTPBearer()


def get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(use_reranker=False)
    return _pipeline


def get_users_db() -> UsersDB:
    """Get or create the users DB singleton."""
    global _users_db
    if _users_db is None:
        _users_db = UsersDB()
    return _users_db


def get_rate_limiter() -> RateLimiter:
    """Get the rate limiter singleton."""
    return _rate_limiter


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> TokenData:
    """
    Validate JWT token and return current user info.
    Raises 401 if token is invalid or expired.
    """
    token_data = decode_access_token(credentials.credentials)

    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user still exists
    users_db = get_users_db()
    user = users_db.get_user(token_data.username)
    if user is None or user.disabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )

    return token_data


async def require_admin(
    current_user: TokenData = Depends(get_current_user),
) -> TokenData:
    """Require admin role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


async def check_rate_limit(
    current_user: TokenData = Depends(get_current_user),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> TokenData:
    """Check rate limit for the current user."""
    if not limiter.is_allowed(current_user.username):
        remaining = limiter.get_reset_time(current_user.username)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry in {remaining:.0f} seconds.",
        )
    return current_user
