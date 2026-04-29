"""
Integration test — requires the API to be running via Docker.

    docker compose up -d api
    pytest api/tests/test_integration.py -v
"""

import os
import pathlib

import pytest
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
TEST_IMAGE = pathlib.Path(__file__).parents[2] / "test_image.jpg"


def test_health_integration():
    resp = requests.get(f"{API_URL}/health", timeout=10)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_integration():
    assert TEST_IMAGE.exists(), f"test_image.jpg not found at {TEST_IMAGE}"
    with open(TEST_IMAGE, "rb") as f:
        resp = requests.post(
            f"{API_URL}/predict",
            files={"file": ("test_image.jpg", f, "image/jpeg")},
            data={
                "latitude": "48.8566",
                "longitude": "2.3522",
                "model_name": "waste-detector-yolov8",
            },
            timeout=60,
        )
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"
    body = resp.json()
    assert "confiance" in body
    assert isinstance(body["confiance"], float)
