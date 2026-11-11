"""Direct test runner for HarvestIQ endpoints and schemas."""

import asyncio
from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import health_check, recommend_crop, get_parameters
from app.schemas import CropRecommendationRequest
from app.utils.validation import validate_crop_inputs
from pydantic import ValidationError


def test_schema_valid():
    payload = {
        "nitrogen": 90.0,
        "phosphorus": 42.0,
        "potassium": 43.0,
        "temperature": 20.87,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.93,
    }
    req = CropRecommendationRequest(**payload)
    assert req.nitrogen == 90.0
    print("  [PASS] test_schema_valid")


def test_schema_invalid_nitrogen():
    try:
        CropRecommendationRequest(
            nitrogen=250.0,  # Invalid: > 200
            phosphorus=42.0,
            potassium=43.0,
            temperature=20.0,
            humidity=80.0,
            ph=6.5,
            rainfall=200.0,
        )
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "nitrogen" in str(e)
        print("  [PASS] test_schema_invalid_nitrogen (422 validation caught)")


def test_schema_invalid_ph():
    try:
        CropRecommendationRequest(
            nitrogen=90.0,
            phosphorus=42.0,
            potassium=43.0,
            temperature=20.0,
            humidity=80.0,
            ph=2.0,  # Invalid: < 3.0
            rainfall=200.0,
        )
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "ph" in str(e)
        print("  [PASS] test_schema_invalid_ph (422 validation caught)")


def test_prediction_service():
    from app.services.prediction import PredictionService, prediction_service
    import tempfile

    req = CropRecommendationRequest(
        nitrogen=90.0,
        phosphorus=42.0,
        potassium=43.0,
        temperature=20.87,
        humidity=82.0,
        ph=6.5,
        rainfall=202.93,
    )

    # 1. Test live prediction with trained model
    rec = prediction_service.predict(req)
    assert rec.success is True
    assert rec.status == "success"
    assert rec.model_status == "ready"
    assert rec.recommended_crop == "Rice"
    assert rec.confidence is not None and rec.confidence > 50.0
    print(f"  [PASS] test_prediction_service_trained: crop={rec.recommended_crop}, conf={rec.confidence}%")

    # 2. Test fallback safeguard when no model directory exists
    with tempfile.TemporaryDirectory() as empty_dir:
        untrained_service = PredictionService(models_dir=Path(empty_dir))
        pending_rec = untrained_service.predict(req)
        assert pending_rec.success is False
        assert pending_rec.status == "model_unavailable"
        assert pending_rec.model_status == "not_trained"
        assert pending_rec.recommended_crop is None
        print(f"  [PASS] test_prediction_service_safeguard: status={pending_rec.status}, model_status={pending_rec.model_status}")


def test_parameter_bounds_coverage():
    from app.utils.validation import PARAMETER_RANGES, validate_crop_inputs
    assert len(PARAMETER_RANGES) == 7
    is_valid, errors = validate_crop_inputs({"nitrogen": 90, "phosphorus": 42, "potassium": 43, "temperature": 20, "humidity": 80, "ph": 6.5, "rainfall": 200})
    assert is_valid is True
    assert len(errors) == 0

    is_invalid, errs = validate_crop_inputs({"nitrogen": -5, "phosphorus": 42, "potassium": 43, "temperature": 20, "humidity": 80, "ph": 6.5, "rainfall": 200})
    assert is_invalid is False
    assert len(errs) == 1
    assert errs[0]["field"] == "nitrogen"
    print("  [PASS] test_parameter_bounds_coverage (bounds check & error formatting)")


if __name__ == "__main__":
    print("=" * 60)
    print(" Running HarvestIQ Verification Tests")
    print("=" * 60)
    test_schema_valid()
    test_schema_invalid_nitrogen()
    test_schema_invalid_ph()
    test_prediction_service()
    test_parameter_bounds_coverage()
    print("=" * 60)
    print(" ALL HARVESTIQ VERIFICATION TESTS PASSED")
    print("=" * 60)
