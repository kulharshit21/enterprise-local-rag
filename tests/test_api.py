"""
API integration tests using httpx.AsyncClient.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from api.main import app


@pytest_asyncio.fixture
async def client():
    """Create test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client):
    """Test the health endpoint returns 200."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test root endpoint returns welcome message."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


@pytest.mark.asyncio
async def test_register_user(client):
    """Test user registration flow."""
    response = await client.post(
        "/auth/register",
        json={
            "username": "test_api_user",
            "password": "testpass123",
            "role": "viewer",
        },
    )
    # May succeed or return 400 if user already exists
    assert response.status_code in [200, 400]

    if response.status_code == 200:
        data = response.json()
        assert "access_token" in data
        assert data["role"] == "viewer"


@pytest.mark.asyncio
async def test_login_user(client):
    """Test login with default user."""
    response = await client.post(
        "/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["role"] == "admin"


@pytest.mark.asyncio
async def test_query_without_auth(client):
    """Test that query endpoint requires authentication."""
    response = await client.post(
        "/query",
        json={"question": "What is the remote work policy?"},
    )
    assert response.status_code in [401, 403]


@pytest.mark.asyncio
async def test_ingest_requires_admin(client):
    """Test that ingestion requires admin role."""
    # Login as viewer
    login_resp = await client.post(
        "/auth/login",
        json={"username": "viewer", "password": "viewer123"},
    )

    if login_resp.status_code == 200:
        token = login_resp.json()["access_token"]

        response = await client.post(
            "/ingest",
            json={"directory": "./data/documents", "role_access": ["viewer"]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403  # Forbidden for non-admin


@pytest.mark.asyncio
async def test_dashboard(client):
    """Test dashboard returns HTML."""
    response = await client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_metrics_summary(client):
    """Test metrics summary endpoint."""
    response = await client.get("/api/metrics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
