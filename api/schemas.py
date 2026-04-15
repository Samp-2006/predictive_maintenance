# ============================================================
# api/schemas.py
# Pydantic v2 request / response models.
# Pydantic validates all data automatically; FastAPI uses these
# to generate the OpenAPI docs and to serialise responses.
# ============================================================

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Single-file prediction result."""
    request_id: str       = Field(..., description="Short unique ID for log tracing")
    file: str             = Field(..., description="Original filename")
    prediction: str       = Field(..., description="'Normal' or 'Faulty'")
    confidence: float     = Field(..., ge=0.0, le=1.0, description="Confidence in predicted class")
    raw_probability: float = Field(..., ge=0.0, le=1.0, description="Raw sigmoid P(Faulty)")

    model_config = {"json_schema_extra": {
        "example": {
            "request_id": "a1b2c3d4",
            "file": "motor_001.wav",
            "prediction": "Faulty",
            "confidence": 0.9732,
            "raw_probability": 0.9732,
        }
    }}


class BatchPredictionResponse(BaseModel):
    """Batch prediction result wrapper."""
    request_id: str
    count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """API liveness probe response."""
    status: str
    model_loaded: bool
    model_path: str


class ModelInfoResponse(BaseModel):
    """Static model metadata."""
    model_name: str
    input_shape: List[int]
    classes: List[str]
    sample_rate: int
    clip_duration_seconds: float
