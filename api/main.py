# ============================================================
# api/main.py
# FastAPI application — entry point for the REST API.
#
# Endpoints:
#   GET  /health          — liveness probe
#   GET  /model/info      — model metadata
#   POST /predict         — single audio file prediction
#   POST /predict/batch   — multiple audio files at once
#
# Production features:
#   • Lifespan event: model loaded once at startup
#   • CORS enabled for the HTML frontend
#   • Structured error responses (RFC 7807-style)
#   • Request/response logging via middleware
#   • File size + extension validation before inference
# ============================================================

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config.settings import api_cfg, model_cfg
from inference.predictor import Predictor
from utils.logger import logger
from api.schemas import PredictionResponse, BatchPredictionResponse, HealthResponse, ModelInfoResponse
from api.middleware import RequestLoggingMiddleware


# ── Application lifespan ──────────────────────────────────────────────────────

predictor = Predictor(model_path=api_cfg.model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before `yield` runs at startup; after `yield` at shutdown.
    Loading the model here ensures it's ready before the first request.
    """
    logger.info("Starting Predictive Maintenance API …")
    try:
        predictor.load()
        logger.success("Model loaded and ready.")
    except FileNotFoundError as exc:
        logger.warning(f"Model not found at startup: {exc}")
        logger.warning("Prediction endpoints will fail until the model is trained.")
    yield
    logger.info("API shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Deep-learning-powered audio analysis for industrial machine health monitoring. "
        "Upload a machine audio clip to receive a Normal / Faulty classification."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (frontend) ───────────────────────────────────────────────────

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ── Validation helper ─────────────────────────────────────────────────────────

def _validate_audio_file(file: UploadFile) -> None:
    """
    Check file extension and size.
    Raises HTTPException(422) on invalid input.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in api_cfg.allowed_audio_extensions:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "UnsupportedFileType",
                "message": f"File type '{suffix}' not supported.",
                "allowed": list(api_cfg.allowed_audio_extensions),
            },
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Liveness probe — returns 200 when the service is running."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        model_path=str(api_cfg.model_path),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info() -> ModelInfoResponse:
    """Returns model configuration metadata."""
    return ModelInfoResponse(
        model_name=model_cfg.model_name,
        input_shape=list(model_cfg.input_shape),
        classes=["Normal", "Faulty"],
        sample_rate=22050,
        clip_duration_seconds=3.0,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Classify a single machine audio recording.

    - **file**: WAV / MP3 / FLAC / OGG audio file (max 10 MB)

    Returns a JSON object with:
    - `prediction`: "Normal" or "Faulty"
    - `confidence`: probability of the predicted class (0–1)
    - `raw_probability`: raw sigmoid output — P(Faulty)
    - `request_id`: unique ID for tracing this request in logs
    """
    _validate_audio_file(file)

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] /predict — file='{file.filename}'")

    try:
        audio_bytes = await file.read()

        # Enforce upload size limit
        max_bytes = api_cfg.max_upload_size_mb * 1024 * 1024
        if len(audio_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "FileTooLarge",
                    "message": f"File exceeds {api_cfg.max_upload_size_mb} MB limit.",
                },
            )

        result = predictor.predict_bytes(audio_bytes, filename=file.filename or "upload.wav")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[{request_id}] Prediction failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PredictionError",
                "message": str(exc),
                "request_id": request_id,
            },
        )

    return PredictionResponse(
        request_id=request_id,
        file=result["file"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        raw_probability=result["raw_probability"],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(files: List[UploadFile] = File(...)) -> BatchPredictionResponse:
    """
    Classify multiple audio files in one request.

    Returns a list of predictions in the same order as the uploaded files.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] /predict/batch — {len(files)} file(s)")

    results = []
    for file in files:
        try:
            _validate_audio_file(file)
            audio_bytes = await file.read()
            result = predictor.predict_bytes(audio_bytes, filename=file.filename or "upload.wav")
            results.append(
                PredictionResponse(
                    request_id=request_id,
                    file=result["file"],
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    raw_probability=result["raw_probability"],
                )
            )
        except Exception as exc:
            logger.warning(f"[{request_id}] Skipping '{file.filename}': {exc}")

    return BatchPredictionResponse(
        request_id=request_id,
        count=len(results),
        predictions=results,
    )
