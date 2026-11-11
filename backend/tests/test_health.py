"""Test suite for GET /health and root endpoints."""

import sys
from pathlib import Path
import sys
import pytest

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    """Verify root endpoint returns HTTP 200 with service information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "HarvestIQ" in data["service"]
    assert data["status"] == "online"


def test_health_endpoint(client):
    """Verify /health endpoint returns HTTP 200 with operational metrics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "dataset_present" in data
    assert "python_version" in data


def test_parameters_endpoint(client):
    """Verify /parameters endpoint returns all 7 parameter bounds."""
    response = client.get("/parameters")
    assert response.status_code == 200
    data = response.json()
    params = data["parameters"]
    assert len(params) == 7
    for key in ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]:
        assert key in params
        assert "min" in params[key]
        assert "max" in params[key]
        assert "unit" in params[key]
