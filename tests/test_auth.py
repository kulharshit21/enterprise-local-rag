"""
Unit tests for authentication and JWT tokens.
"""

import pytest
from security.auth import (
    hash_password, verify_password,
    create_access_token, decode_access_token,
)


class TestPasswordHashing:
    """Tests for password hashing and verification."""

    def test_hash_password(self):
        """Test that hashing produces a non-plaintext result."""
        hashed = hash_password("test123")
        assert hashed != "test123"
        assert len(hashed) > 20

    def test_verify_correct_password(self):
        """Test that correct password verifies."""
        hashed = hash_password("secure_password")
        assert verify_password("secure_password", hashed) is True

    def test_verify_wrong_password(self):
        """Test that wrong password does not verify."""
        hashed = hash_password("correct_password")
        assert verify_password("wrong_password", hashed) is False

    def test_different_hashes_same_password(self):
        """Test that same password produces different hashes (salt)."""
        h1 = hash_password("same_password")
        h2 = hash_password("same_password")
        assert h1 != h2  # Different salts


class TestJWT:
    """Tests for JWT token creation and validation."""

    def test_create_and_decode_token(self):
        """Test token creation and decoding round-trip."""
        token = create_access_token({"sub": "testuser", "role": "researcher"})
        data = decode_access_token(token)

        assert data is not None
        assert data.username == "testuser"
        assert data.role == "researcher"

    def test_invalid_token(self):
        """Test that invalid token returns None."""
        data = decode_access_token("this.is.not.a.valid.token")
        assert data is None

    def test_token_contains_role(self):
        """Test that role is embedded in token."""
        token = create_access_token({"sub": "admin_user", "role": "admin"})
        data = decode_access_token(token)
        assert data.role == "admin"

    def test_empty_token(self):
        """Test handling of empty token string."""
        data = decode_access_token("")
        assert data is None
