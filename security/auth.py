"""
JWT authentication and password hashing.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel

from config import settings


class UserModel(BaseModel):
    """User data model."""
    username: str
    hashed_password: str
    role: str = "viewer"
    email: str = ""
    disabled: bool = False


class TokenData(BaseModel):
    """JWT token payload."""
    username: str
    role: str
    exp: Optional[datetime] = None


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    salt = bcrypt.gensalt(rounds=4)  # Faster rounds for dev/demo
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def create_access_token(data: Dict[str, Any], expires_delta: timedelta = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Dict with 'sub' (username) and 'role'
        expires_delta: Token expiry duration

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT token.

    Returns TokenData if valid, None if invalid/expired.
    """
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
        username = payload.get("sub")
        role = payload.get("role", "viewer")

        if username is None:
            return None

        return TokenData(username=username, role=role)
    except JWTError:
        return None
