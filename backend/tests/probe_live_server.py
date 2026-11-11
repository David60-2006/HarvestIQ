"""Starts an in-process live Uvicorn server, sends real HTTP requests, and verifies responses."""

import json
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
import uvicorn


def run_probe():
    print("=" * 70)
    print(" PROBING LIVE FASTAPI SERVER VIA HTTP (URLLIB)")
    print("=" * 70)

    config = uvicorn.Config(app=app, host="127.0.0.1", port=8765, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run)
    t.daemon = True
    t.start()
    time.sleep(1.5)

    base_url = "http://127.0.0.1:8765"

    try:
        # 1. GET /health
        print("\n[HTTP PROBE 1] Probing GET /health ...")
        with urllib.request.urlopen(f"{base_url}/health") as resp:
            health_bytes = resp.read()
            health_json = json.loads(health_bytes.decode("utf-8"))
            print(f"  Status Code: {resp.status}")
            print(f"  Response: {health_json}")
            assert resp.status == 200
            assert health_json["status"] == "healthy"
            assert health_json["model_loaded"] is True

        # 2. GET /model-info
        print("\n[HTTP PROBE 2] Probing GET /model-info ...")
        with urllib.request.urlopen(f"{base_url}/model-info") as resp:
            info_bytes = resp.read()
            info_json = json.loads(info_bytes.decode("utf-8"))
            print(f"  Status Code: {resp.status}")
            print(f"  Model Name: {info_json['model_name']}")
            print(f"  Version: {info_json['model_version']}")
            print(f"  Supported Crops: {len(info_json['supported_crops'])} classes")
            print(f"  Score Type: {info_json['score_type']}")
            assert resp.status == 200
            assert info_json["model_name"] == "RandomForestClassifier"

        # 3. GET /docs (Swagger)
        print("\n[HTTP PROBE 3] Probing GET /docs (Swagger UI) ...")
        with urllib.request.urlopen(f"{base_url}/docs") as resp:
            docs_html = resp.read().decode("utf-8")
            print(f"  Status Code: {resp.status} (Swagger UI reachable)")
            assert resp.status == 200
            assert "swagger-ui" in docs_html.lower() or "openapi" in docs_html.lower()

        # 4. POST /recommend (Valid)
        print("\n[HTTP PROBE 4] Probing POST /recommend (Valid Rice Input) ...")
        payload = {
            "nitrogen": 90.0,
            "phosphorus": 42.0,
            "potassium": 43.0,
            "temperature": 20.87,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 202.93,
        }
        req_data = json.dumps(payload).encode("utf-8")
        post_req = urllib.request.Request(
            f"{base_url}/recommend",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(post_req) as resp:
            rec_bytes = resp.read()
            rec_json = json.loads(rec_bytes.decode("utf-8"))
            print(f"  Status Code: {resp.status}")
            print(f"  Response Body:\n{json.dumps(rec_json, indent=2)}")
            assert resp.status == 200
            assert rec_json["success"] is True
            assert rec_json["recommended_crop"] == "Rice"
            assert len(rec_json["recommendations"]) == 3
            assert rec_json["recommendations"][0]["crop"] == "Rice"
            assert rec_json["recommendations"][0]["score"] == 0.9

        # 5. POST /recommend (Invalid - triggers 422)
        print("\n[HTTP PROBE 5] Probing POST /recommend (Invalid Nitrogen=250) ...")
        invalid_payload = dict(payload)
        invalid_payload["nitrogen"] = 250.0
        inv_data = json.dumps(invalid_payload).encode("utf-8")
        inv_req = urllib.request.Request(
            f"{base_url}/recommend",
            data=inv_data,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(inv_req)
            assert False, "Expected HTTP 422 error"
        except urllib.error.HTTPError as e:
            err_body = json.loads(e.read().decode("utf-8"))
            print(f"  Caught Expected HTTP {e.code}: {err_body}")
            assert e.code == 422

        print("\n" + "=" * 70)
        print(" ALL HTTP SERVER PROBES COMPLETED SUCCESSFULLY")
        print("=" * 70)

    finally:
        server.should_exit = True


if __name__ == "__main__":
    run_probe()
