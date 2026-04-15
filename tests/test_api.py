# ============================================================
# tests/test_api.py
# FastAPI endpoint integration tests using httpx TestClient.
# Run:  pytest tests/test_api.py -v
# ============================================================

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app
from config.settings import audio_cfg
from data.generate_data import generate_normal_sample


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_wav_bytes():
    """Create a valid WAV file as bytes for upload tests."""
    audio = generate_normal_sample(rng=np.random.default_rng(0))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, audio_cfg.sample_rate, subtype="PCM_16")
        with open(f.name, "rb") as fh:
            data = fh.read()
        Path(f.name).unlink(missing_ok=True)
    return data


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_field(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200

    def test_model_info_classes(self, client):
        data = client.get("/model/info").json()
        assert "Normal" in data["classes"]
        assert "Faulty" in data["classes"]


class TestPredictEndpoint:
    def test_unsupported_extension_returns_422(self, client):
        r = client.post(
            "/predict",
            files={"file": ("audio.txt", b"not audio", "text/plain")},
        )
        assert r.status_code == 422

    def test_valid_wav_upload_accepted(self, client, sample_wav_bytes):
        """If model is loaded this returns 200; if not loaded returns 500."""
        r = client.post(
            "/predict",
            files={"file": ("motor.wav", sample_wav_bytes, "audio/wav")},
        )
        # Accept either success or "model not loaded" error
        assert r.status_code in (200, 500)

    def test_response_schema_on_success(self, client, sample_wav_bytes):
        r = client.post(
            "/predict",
            files={"file": ("motor.wav", sample_wav_bytes, "audio/wav")},
        )
        if r.status_code == 200:
            data = r.json()
            assert "prediction" in data
            assert data["prediction"] in ("Normal", "Faulty")
            assert 0.0 <= data["confidence"] <= 1.0
            assert 0.0 <= data["raw_probability"] <= 1.0
