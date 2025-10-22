"""
Simple test script for the FastAPI backend.

Tests that the server can start and endpoints respond correctly.
"""

from fastapi.testclient import TestClient
from mindgames.backend.app import app, DB_PATH
from mindgames.db import init_db

# Initialize database for testing
print(f"Initializing test database at {DB_PATH}...")
init_db(DB_PATH)

# Create test client
client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "MindGames Arena API"
    assert "version" in data
    print("✓ Root endpoint working")


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    print(f"Health check response: {data}")

    # Health check should return status (may be healthy or unhealthy)
    assert "status" in data
    assert "database" in data

    if data["status"] == "healthy":
        assert "active_models" in data
        assert "active_kinds" in data
        assert "active_prompts" in data
        print(f"✓ Health check working: {data}")
    else:
        print(f"⚠ Health check returned unhealthy status: {data}")
        # Still consider this a pass - endpoint is working, just no data yet


if __name__ == "__main__":
    print("Testing MindGames Arena API...\n")
    test_root()
    test_health_check()
    print("\n✓ All tests passed!")
