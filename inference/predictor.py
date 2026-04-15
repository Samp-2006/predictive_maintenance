# ============================================================
# inference/predictor.py
# Production inference pipeline.
#
# Accepts a raw audio file path or bytes, runs the full
# preprocessing → feature extraction → model prediction chain,
# and returns a structured result dict.
#
# The Predictor class is a singleton-friendly service object:
#   - Model is loaded once at startup (heavy operation)
#   - predict() is a lightweight call (ms-level latency)
#   - Thread-safe for concurrent FastAPI requests
# ============================================================

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import soundfile as sf

from config.settings import api_cfg, audio_cfg, model_cfg
from models.cnn_model import load_model
from preprocessing.audio_loader import AudioLoader
from preprocessing.feature_extractor import FeatureExtractor
from utils.logger import logger
from utils.helpers import label_from_probability, timeit


class Predictor:
    """
    Stateful inference engine that holds the loaded model in memory.

    Usage (FastAPI startup):
        predictor = Predictor()           # loads model once
        result = predictor.predict(path)  # cheap inference
    """

    def __init__(self, model_path: str | Path = api_cfg.model_path) -> None:
        self.model_path = Path(model_path)
        self._model = None  # lazy load — avoids crash if model doesn't exist yet

        self._loader    = AudioLoader(
            target_sr=audio_cfg.sample_rate,
            duration=audio_cfg.duration,
        )
        self._extractor = FeatureExtractor(
            target_size=(model_cfg.input_shape[0], model_cfg.input_shape[1])
        )

    # ── Model management ─────────────────────────────────────────────────

    def load(self) -> None:
        """Explicitly load the model.  Called once during API startup."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{self.model_path}'.\n"
                "Run: python training/train.py"
            )
        self._model = load_model(str(self.model_path))
        logger.info(f"Predictor ready — model loaded from '{self.model_path}'")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Core prediction ──────────────────────────────────────────────────

    @timeit
    def predict(self, audio_path: str | Path) -> Dict[str, Any]:
        """
        End-to-end prediction from file path.

        Args:
            audio_path: Path to audio file (WAV / MP3 / FLAC / OGG).

        Returns:
            {
                "prediction": "Normal" | "Faulty",
                "confidence": 0.97,          # confidence in predicted class
                "raw_probability": 0.972,    # P(Faulty) from sigmoid
                "file": "motor_001.wav"
            }
        """
        if not self.is_loaded:
            self.load()

        audio_path = Path(audio_path)

        # 1. Load waveform
        waveform, _ = self._loader.load(audio_path)

        # 2. Extract CNN feature
        feature = self._extractor.extract_cnn_input(waveform)   # (H, W, 1)
        batch   = feature[np.newaxis, ...]                       # (1, H, W, 1)

        # 3. Forward pass
        raw_prob = float(self._model.predict(batch, verbose=0)[0, 0])

        # 4. Decode result
        result = label_from_probability(raw_prob)
        result["file"] = audio_path.name

        logger.info(
            f"[Predictor] {audio_path.name} → {result['prediction']} "
            f"({result['confidence']*100:.1f}% confidence)"
        )
        return result

    @timeit
    def predict_bytes(self, audio_bytes: bytes, filename: str = "upload.wav") -> Dict[str, Any]:
        """
        Predict directly from in-memory bytes (for API file uploads).

        Writes bytes to a temp file, runs predict(), then cleans up.
        """
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        try:
            result = self.predict(tmp_path)
            result["file"] = filename   # overwrite temp name with original
        finally:
            tmp_path.unlink(missing_ok=True)

        return result

    # ── Batch inference ──────────────────────────────────────────────────

    def predict_batch(self, audio_paths: list[str | Path]) -> list[Dict[str, Any]]:
        """
        Predict on a list of files in one batched forward pass.
        More efficient than calling predict() N times for large batches.
        """
        if not self.is_loaded:
            self.load()

        features, names = [], []
        for p in audio_paths:
            try:
                wav, _ = self._loader.load(p)
                feat   = self._extractor.extract_cnn_input(wav)
                features.append(feat)
                names.append(Path(p).name)
            except Exception as exc:
                logger.warning(f"Skipping '{p}': {exc}")

        if not features:
            return []

        batch    = np.stack(features, axis=0)                       # (N, H, W, 1)
        probs    = self._model.predict(batch, verbose=0).flatten()  # (N,)

        return [
            {**label_from_probability(float(p)), "file": n}
            for p, n in zip(probs, names)
        ]
