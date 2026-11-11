"""Comprehensive integration and regression tests for HarvestIQ Prompt 5 API.
Uses direct synchronous coroutine resolution to prevent Windows sandbox IOCP blocking.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app, health_check, get_model_info, recommend_crop, get_parameters
from app.schemas import CropRecommendationRequest
from app.services.prediction import PredictionService, prediction_service
from pydantic import ValidationError


def run_sync(coro):
    """Executes a non-blocking coroutine synchronously without event loop overhead."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    raise RuntimeError("Coroutine did not finish synchronously")


def run_prompt5_api_tests():
    print("=" * 70)
    print(" HARVESTIQ: PROMPT 5 FASTAPI PREDICTION API VERIFICATION SUITE")
    print("=" * 70)

    # -----------------------------------------------------------------
    # TEST 1: GET /health
    # -----------------------------------------------------------------
    print("\n[TEST 1] Verifying GET /health endpoint...")
    data_health = run_sync(health_check())
    assert data_health.status == "healthy"
    assert data_health.model_loaded is True
    assert data_health.model_name == "RandomForestClassifier"
    assert data_health.version == "1.0.0"
    print(f"  [PASS] GET /health (HTTP 200 equivalent)")
    print(f"         Status: {data_health.status} | Model Loaded: {data_health.model_loaded} | Model: {data_health.model_name}")

    # -----------------------------------------------------------------
    # TEST 2: GET /model-info
    # -----------------------------------------------------------------
    print("\n[TEST 2] Verifying GET /model-info endpoint...")
    data_info = run_sync(get_model_info())
    assert data_info.model_name == "RandomForestClassifier"
    assert data_info.model_version == "1.0.0"
    assert data_info.is_loaded is True
    assert data_info.score_type == "uncalibrated model score"
    assert data_info.classes_count == 22
    assert "rice" in data_info.supported_crops
    assert "cotton" in data_info.supported_crops
    assert data_info.evaluation_metrics is not None
    assert data_info.evaluation_metrics["Test_Accuracy"] == 0.9955
    print(f"  [PASS] GET /model-info (HTTP 200 equivalent)")
    print(f"         Algorithm: {data_info.model_name} v{data_info.model_version}")
    print(f"         Classes Supported: {data_info.classes_count} crops ({data_info.supported_crops[:5]}...)")
    print(f"         Holdout Test Accuracy: {data_info.evaluation_metrics['Test_Accuracy']*100:.2f}%")
    print(f"         5-Fold CV Macro F1:    {data_info.evaluation_metrics['CV_Macro_F1_Mean']*100:.2f}%")

    # -----------------------------------------------------------------
    # TEST 3: POST /recommend with valid input
    # -----------------------------------------------------------------
    print("\n[TEST 3] Verifying POST /recommend with valid agro-climatic input...")
    rice_payload = {
        "nitrogen": 90.0,
        "phosphorus": 42.0,
        "potassium": 43.0,
        "temperature": 20.87,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.93,
    }
    req_rice = CropRecommendationRequest(**rice_payload)
    json_resp = run_sync(recommend_crop(req_rice))
    data_rec = json.loads(json_resp.body.decode("utf-8"))

    assert json_resp.status_code == 200
    assert data_rec["success"] is True
    assert data_rec["recommended_crop"] == "Rice"
    assert data_rec["model_name"] == "RandomForestClassifier"
    assert data_rec["model_version"] == "1.0.0"
    assert data_rec["score_type"] == "uncalibrated model score"
    assert len(data_rec["recommendations"]) == 3

    # Verify rank 1
    r1 = data_rec["recommendations"][0]
    assert r1["rank"] == 1
    assert r1["crop"] == "Rice"
    assert r1["score"] == 0.9
    assert r1["confidence_percentage"] == 90.0

    # Verify rank 2
    r2 = data_rec["recommendations"][1]
    assert r2["rank"] == 2
    assert r2["crop"] == "Jute"
    assert r2["score"] == 0.1
    assert r2["confidence_percentage"] == 10.0

    print(f"  [PASS] POST /recommend (HTTP {json_resp.status_code})")
    print(f"         Primary Recommendation: {data_rec['recommended_crop']} ({r1['confidence_percentage']}%)")
    print(f"         Ranked Candidates: {[(r['rank'], r['crop'], r['confidence_percentage']) for r in data_rec['recommendations']]}")
    print(f"         Score Type: '{data_rec['score_type']}' | Version: {data_rec['model_version']}")

    # -----------------------------------------------------------------
    # TEST 4: POST /recommend with invalid input (HTTP 422 check)
    # -----------------------------------------------------------------
    print("\n[TEST 4] Verifying input validation boundaries (HTTP 422 trigger)...")
    invalid_cases = [
        {"field": "nitrogen", "val": 250.0, "reason": "exceeds max 200"},
        {"field": "nitrogen", "val": -5.0, "reason": "below min 0"},
        {"field": "potassium", "val": 350.0, "reason": "exceeds max 300"},
        {"field": "temperature", "val": 55.0, "reason": "exceeds max 50"},
        {"field": "humidity", "val": 105.0, "reason": "exceeds max 100"},
        {"field": "ph", "val": 2.5, "reason": "below min 3.0"},
        {"field": "ph", "val": 10.5, "reason": "exceeds max 10.0"},
        {"field": "rainfall", "val": 450.0, "reason": "exceeds max 400"},
    ]

    for tc in invalid_cases:
        p = dict(rice_payload)
        p[tc["field"]] = tc["val"]
        try:
            CropRecommendationRequest(**p)
            assert False, f"Should have failed validation for {tc['field']}={tc['val']}"
        except ValidationError as e:
            assert tc["field"] in str(e)
            print(f"  [PASS] Caught invalid {tc['field']}={tc['val']} ({tc['reason']}) -> HTTP 422 response")

    # -----------------------------------------------------------------
    # TEST 5: Missing model file handling
    # -----------------------------------------------------------------
    print("\n[TEST 5] Verifying missing model file error handling (HTTP 503 safeguard)...")
    with tempfile.TemporaryDirectory() as empty_dir:
        mock_service = PredictionService(models_dir=Path(empty_dir))
        assert mock_service.is_model_loaded() is False

        mock_res = mock_service.predict(req_rice)
        assert mock_res.success is False
        assert mock_res.status == "model_unavailable"
        assert mock_res.model_status == "not_trained"
        assert mock_res.recommended_crop is None
        print(f"  [PASS] Missing model handled gracefully: success={mock_res.success}, status='{mock_res.status}', model_status='{mock_res.model_status}'")

    # -----------------------------------------------------------------
    # TEST 6: Model prediction consistency across diverse agro-climates
    # -----------------------------------------------------------------
    print("\n[TEST 6] Verifying prediction consistency across diverse crop classes...")
    benchmarks = [
        {
            "crop": "Cotton",
            "payload": {"nitrogen": 118.0, "phosphorus": 46.0, "potassium": 19.0, "temperature": 23.97, "humidity": 81.42, "ph": 6.96, "rainfall": 69.84},
        },
        {
            "crop": "Coffee",
            "payload": {"nitrogen": 104.0, "phosphorus": 18.0, "potassium": 30.0, "temperature": 23.60, "humidity": 60.39, "ph": 6.78, "rainfall": 140.94},
        },
        {
            "crop": "Chickpea",
            "payload": {"nitrogen": 40.0, "phosphorus": 72.0, "potassium": 77.0, "temperature": 17.02, "humidity": 16.98, "ph": 7.48, "rainfall": 88.55},
        },
    ]

    for bm in benchmarks:
        req_bm = CropRecommendationRequest(**bm["payload"])
        resp_bm = run_sync(recommend_crop(req_bm))
        res_json = json.loads(resp_bm.body.decode("utf-8"))
        assert res_json["recommended_crop"] == bm["crop"]

        # Re-run to verify deterministic consistency
        resp_repeat = run_sync(recommend_crop(req_bm))
        res_rep_json = json.loads(resp_repeat.body.decode("utf-8"))
        assert res_rep_json["recommended_crop"] == bm["crop"]
        print(f"  [PASS] Benchmark {bm['crop']}: Predicted '{res_json['recommended_crop']}' (Score: {res_json['recommendations'][0]['score']}, 100% Deterministic)")

    print("\n" + "=" * 70)
    print(" ALL 6 PROMPT 5 FASTAPI TEST SCENARIOS PASSED WITH ZERO ERRORS")
    print("=" * 70)


if __name__ == "__main__":
    run_prompt5_api_tests()
