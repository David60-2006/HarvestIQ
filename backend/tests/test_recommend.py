"""Test suite for POST /recommend endpoint and agro-climatic parameter validation."""

import sys
from pathlib import Path
import pytest

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "nitrogen": 90.0,
    "phosphorus": 42.0,
    "potassium": 43.0,
    "temperature": 20.87,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.93,
}


def test_recommend_valid_input(client):
    """Verify POST /recommend succeeds with HTTP 200 for valid parameters."""
    response = client.post("/recommend", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "status" in data
    assert "validated_input" in data
    assert data["validated_input"]["nitrogen"] == 90.0


def test_recommend_nitrogen_lower_bound(client):
    """Nitrogen < 0 must return HTTP 422."""
    payload = {**VALID_PAYLOAD, "nitrogen": -5.0}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert any(err["field"] == "nitrogen" for err in data["errors"])


def test_recommend_nitrogen_upper_bound(client):
    """Nitrogen > 200 must return HTTP 422."""
    payload = {**VALID_PAYLOAD, "nitrogen": 205.0}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert any(err["field"] == "nitrogen" for err in data["errors"])


def test_recommend_phosphorus_upper_bound(client):
    """Phosphorus > 200 must return HTTP 422."""
    payload = {**VALID_PAYLOAD, "phosphorus": 210.0}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any(err["field"] == "phosphorus" for err in data["errors"])


def test_recommend_potassium_upper_bound(client):
    """Potassium > 300 must return HTTP 422."""
    payload = {**VALID_PAYLOAD, "potassium": 350.0}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any(err["field"] == "potassium" for err in data["errors"])


def test_recommend_temperature_bounds(client):
    """Temperature < 0 or > 50 must return HTTP 422."""
    payload_low = {**VALID_PAYLOAD, "temperature": -1.0}
    response = client.post("/recommend", json=payload_low)
    assert response.status_code == 422

    payload_high = {**VALID_PAYLOAD, "temperature": 55.0}
    response = client.post("/recommend", json=payload_high)
    assert response.status_code == 422


def test_recommend_humidity_bounds(client):
    """Humidity < 0 or > 100 must return HTTP 422."""
    payload = {**VALID_PAYLOAD, "humidity": 105.0}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422


def test_recommend_ph_bounds(client):
    """pH < 3 or > 10 must return HTTP 422."""
    payload_acidic = {**VALID_PAYLOAD, "ph": 2.5}
    response = client.post("/recommend", json=payload_acidic)
    assert response.status_code == 422

    payload_alkaline = {**VALID_PAYLOAD, "ph": 11.0}
    response = client.post("/recommend", json=payload_alkaline)
    assert response.status_code == 422


def test_recommend_rainfall_bounds(client):
    """Rainfall < 0 or > 400 must return HTTP 422."""
    payload = {**VALID_PAYLOAD, "rainfall": 450.0}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422


def test_recommend_missing_field(client):
    """Missing required parameters must return HTTP 422 with field notice."""
    incomplete_payload = {
        "nitrogen": 90.0,
        "phosphorus": 42.0,
        # missing potassium, temperature, humidity, ph, rainfall
    }
    response = client.post("/recommend", json=incomplete_payload)
    assert response.status_code == 422
    data = response.json()
    fields = [err["field"] for err in data["errors"]]
    assert "potassium" in fields
