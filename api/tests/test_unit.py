"""
Unit tests for the Waste Detection API.

Run without Docker — MLflow HTTP calls are mocked, DB uses a temp file.
"""

import io
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Point DB to a temp file so tests don't touch real data
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DB_PATH"] = _tmp_db.name
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import main  # noqa: E402  (import after env vars are set)


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """Session-scoped TestClient — triggers startup (DB init) exactly once."""
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(autouse=True)
def clear_model_cache():
    main._registered_model_names.cache_clear()
    yield
    main._registered_model_names.cache_clear()


@pytest.fixture()
def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def mock_registry():
    """Mock MLflow HTTP to return a single known model."""
    payload = {
        "registered_models": [
            {
                "name": "waste-detector-yolov8",
                "tags": [{"key": "framework", "value": "other"}],  # skip real YOLO in unit tests
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = payload

    with patch("main.requests.get", return_value=mock_resp):
        yield


# ── Tests ────────────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "timestamp" in body


def test_predict_valid_returns_confiance(client, jpeg_bytes, mock_registry):
    """Valid request returns 200 with a float confiance field."""
    resp = client.post(
        "/predict",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"latitude": "48.8566", "longitude": "2.3522", "model_name": "waste-detector-yolov8"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "confiance" in body
    assert isinstance(body["confiance"], float)
    assert "model_name" in body
    assert "timestamp" in body


def test_predict_invalid_latitude_returns_422(client, jpeg_bytes, mock_registry):
    resp = client.post(
        "/predict",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"latitude": "999", "longitude": "2.3522", "model_name": "waste-detector-yolov8"},
    )
    assert resp.status_code == 422


def test_predict_invalid_mime_returns_422(client, mock_registry):
    resp = client.post(
        "/predict",
        files={"file": ("doc.txt", b"not an image", "text/plain")},
        data={"latitude": "48.8566", "longitude": "2.3522", "model_name": "waste-detector-yolov8"},
    )
    assert resp.status_code == 422


def test_predict_unknown_model_returns_422(client, jpeg_bytes, mock_registry):
    resp = client.post(
        "/predict",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"latitude": "48.8566", "longitude": "2.3522", "model_name": "nonexistent-model"},
    )
    assert resp.status_code == 422


def test_history_returns_list(client, jpeg_bytes, mock_registry):
    # Ensure at least one prediction exists
    client.post(
        "/predict",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"latitude": "48.8566", "longitude": "2.3522", "model_name": "waste-detector-yolov8"},
    )
    resp = client.get("/history")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    assert len(resp.json()) >= 1


def test_metrics_endpoint_returns_prometheus_text(client):
    """GET /metrics returns 200 with all required metric names."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "ml_predictions_total" in resp.text
    assert "ml_inference_latency_seconds" in resp.text
    assert "ml_predictions_by_model_total" in resp.text
    assert "ml_validation_errors_total" in resp.text


def test_predict_writes_jsonl_log(client, jpeg_bytes, mock_registry, tmp_path):
    """POST /predict appends a valid JSON line to LOG_PATH."""
    import json as _json

    log_file = tmp_path / "predictions.jsonl"
    original = main.LOG_PATH
    main.LOG_PATH = str(log_file)
    try:
        resp = client.post(
            "/predict",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={"latitude": "48.8566", "longitude": "2.3522", "model_name": "waste-detector-yolov8"},
        )
        assert resp.status_code == 200
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        entry = _json.loads(lines[-1])
        assert "timestamp" in entry
        assert "confiance" in entry
        assert "latence_ms" in entry
        assert entry["source"] == "manual"
        assert entry["latitude"] == 48.8566
    finally:
        main.LOG_PATH = original
