"""Test configuration and fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def client():
    """Create test client."""
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_workspace_data():
    """Sample workspace creation data."""
    return {
        "name": "Test Workspace",
        "slug": "test-workspace",
        "description": "A test workspace",
    }


@pytest.fixture
def sample_api_key_data():
    """Sample API key creation data."""
    return {
        "name": "Test Key",
        "expires_days": 30,
    }
