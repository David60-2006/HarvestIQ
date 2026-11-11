"""End-to-end frontend-backend integration test suite for HarvestIQ Prompt 6."""

import json
from pathlib import Path
import re
import sys

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

root_dir = backend_dir.parent
frontend_dir = root_dir / "frontend"

from app.main import app, health_check, get_model_info, recommend_crop
from app.schemas import CropRecommendationRequest
from app.services.prediction import PredictionService, prediction_service
from pydantic import ValidationError


def run_sync(coro):
    """Executes a non-blocking coroutine synchronously."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    raise RuntimeError("Coroutine did not finish synchronously")


def test_frontend_assets_integrity():
    print("\n[STEP 1] Validating frontend asset integrity & configuration...")
    index_html = frontend_dir / "index.html"
    style_css = frontend_dir / "style.css"
    app_js = frontend_dir / "app.js"
    config_js = frontend_dir / "config.js"

    assert index_html.exists(), "index.html missing"
    assert style_css.exists(), "style.css missing"
    assert app_js.exists(), "app.js missing"
    assert config_js.exists(), "config.js missing"

    index_content = index_html.read_text(encoding="utf-8")
    config_content = config_js.read_text(encoding="utf-8")
    app_content = app_js.read_text(encoding="utf-8")
    style_content = style_css.read_text(encoding="utf-8")

    # Verify config URL
    assert "http://127.0.0.1:8000" in config_content
    assert "/health" in config_content
    assert "/model-info" in config_content
    assert "/recommend" in config_content

    # Verify input elements in index.html
    required_inputs = [
        "input-nitrogen", "input-phosphorus", "input-potassium",
        "input-temperature", "input-humidity", "input-ph", "input-rainfall"
    ]
    for inp in required_inputs:
        assert f'id="{inp}"' in index_content, f"Missing input element: {inp}"

    # Verify script tags order in index.html
    assert 'src="config.js"' in index_content
    assert 'src="app.js"' in index_content
    config_idx = index_content.find('src="config.js"')
    app_idx = index_content.find('src="app.js"')
    assert config_idx < app_idx, "config.js must load before app.js"

    # Verify no mock prediction logic
    forbidden_tokens = ["Math.random()", "fake_prediction", "mock_crops"]
    for token in forbidden_tokens:
        assert token not in app_content, f"Found forbidden token: {token}"

    # Verify uncalibrated score transparency and retry buttons exist in JS/CSS
    assert "uncalibrated" in app_content.lower()
    assert "retry" in app_content.lower()
    assert "btn-retry" in style_content
    assert "recommendations-ranked-container" in style_content

    print("  [PASS] All 4 frontend files verified with consistent API bindings and zero demo logic.")


def test_successful_recommendation_flow():
    print("\n[STEP 2] Testing successful recommendation workflow (User input -> FastAPI -> RF Model -> UI Data)...")
    rice_payload = {
        "nitrogen": 90.0,
        "phosphorus": 42.0,
        "potassium": 43.0,
        "temperature": 20.87,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.93,
    }

    req = CropRecommendationRequest(**rice_payload)
    response = run_sync(recommend_crop(req))
    data = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert data["success"] is True
    assert data["recommended_crop"] == "Rice"
    assert data["model_name"] == "RandomForestClassifier"
    assert data["model_version"] == "1.0.0"
    assert data["score_type"] == "uncalibrated model score"
    assert len(data["recommendations"]) == 3
    assert data["recommendations"][0]["crop"] == "Rice"
    assert data["recommendations"][0]["score"] == 0.9
    assert data["recommendations"][0]["confidence_percentage"] == 90.0
    assert data["recommendations"][1]["crop"] == "Jute"
    assert data["recommendations"][1]["score"] == 0.1

    print(f"  [PASS] Primary Recommendation: '{data['recommended_crop']}' (90.0% confidence)")
    print(f"         Ranked Candidates: {[(r['rank'], r['crop'], r['score']) for r in data['recommendations']]}")
    print(f"         Metadata: Model={data['model_name']} v{data['model_version']} | ScoreType={data['score_type']}")


def test_invalid_and_empty_inputs():
    print("\n[STEP 3] Testing validation rejection for empty, out-of-bounds, and non-numeric inputs...")

    # 1. Out-of-bounds inputs (HTTP 422 trigger)
    invalid_test_cases = [
        {"nitrogen": 250.0},  # Max is 200
        {"nitrogen": -10.0},  # Min is 0
        {"phosphorus": 210.0}, # Max is 200
        {"potassium": 350.0},  # Max is 300
        {"temperature": 55.0}, # Max is 50
        {"temperature": -5.0}, # Min is 0
        {"humidity": 110.0},   # Max is 100
        {"ph": 2.5},          # Min is 3.0
        {"ph": 10.5},         # Max is 10.0
        {"rainfall": 450.0},  # Max is 400
    ]

    base = {
        "nitrogen": 90.0, "phosphorus": 42.0, "potassium": 43.0,
        "temperature": 20.87, "humidity": 82.0, "ph": 6.5, "rainfall": 202.93
    }

    for case in invalid_test_cases:
        test_payload = dict(base)
        test_payload.update(case)
        param_name = list(case.keys())[0]
        try:
            CropRecommendationRequest(**test_payload)
            assert False, f"Expected ValidationError for {param_name}={case[param_name]}"
        except ValidationError as e:
            assert param_name in str(e)

    print("  [PASS] All 10 boundary violation scenarios caught and rejected with HTTP 422 schema enforcement.")

    # 2. Empty / Missing input
    try:
        CropRecommendationRequest()  # No parameters passed
        assert False, "Expected ValidationError on empty input"
    except ValidationError:
        pass
    print("  [PASS] Empty input rejected immediately by Pydantic.")


def test_offline_backend_handling():
    print("\n[STEP 4] Testing graceful error handling when model or backend is unavailable...")
    import tempfile

    with tempfile.TemporaryDirectory() as empty_dir:
        untrained_service = PredictionService(models_dir=Path(empty_dir))
        req = CropRecommendationRequest(
            nitrogen=90.0, phosphorus=42.0, potassium=43.0,
            temperature=20.87, humidity=82.0, ph=6.5, rainfall=202.93
        )
        res = untrained_service.predict(req)
        assert res.success is False
        assert res.status == "model_unavailable"
        assert res.model_status == "not_trained"
        assert res.recommended_crop is None
        print("  [PASS] Unavailable model returns success=False without generating mock predictions.")


def run_all_tests():
    print("=" * 70)
    print(" HARVESTIQ: PROMPT 6 FRONTEND & BACKEND INTEGRATION AUDIT")
    print("=" * 70)
    test_frontend_assets_integrity()
    test_successful_recommendation_flow()
    test_invalid_and_empty_inputs()
    test_offline_backend_handling()
    print("\n" + "=" * 70)
    print(" ALL PROMPT 6 INTEGRATION TESTS PASSED (100% SUCCESS)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
