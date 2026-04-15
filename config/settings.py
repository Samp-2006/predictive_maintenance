# ============================================================
# config/settings.py
# Central configuration for the entire system.
# All tuneable parameters live here — no magic numbers in code.
# ============================================================

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Tuple


class AudioConfig(BaseSettings):
    """Audio signal parameters used during synthesis and feature extraction."""

    sample_rate: int = 22050          # Hz — standard for audio ML tasks
    duration: float = 3.0             # seconds per clip
    n_fft: int = 2048                 # FFT window size for spectrogram
    hop_length: int = 512             # Samples between successive frames
    n_mels: int = 128                 # Mel filterbank bins
    n_mfcc: int = 40                  # Number of MFCC coefficients
    fmin: float = 20.0                # Min frequency for mel filterbank (Hz)
    fmax: float = 8000.0              # Max frequency for mel filterbank (Hz)

    class Config:
        env_prefix = "AUDIO_"


class DataConfig(BaseSettings):
    """Paths and dataset split configuration."""

    data_dir: Path = Path("data")
    normal_dir: Path = Path("data/normal")
    faulty_dir: Path = Path("data/faulty")
    saved_models_dir: Path = Path("saved_models")
    logs_dir: Path = Path("logs")

    # Number of synthetic samples per class
    n_normal_samples: int = 500
    n_faulty_samples: int = 500

    # Train / validation / test split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    random_seed: int = 42

    class Config:
        env_prefix = "DATA_"


class ModelConfig(BaseSettings):
    """CNN model hyper-parameters."""

    input_shape: Tuple[int, int, int] = (128, 128, 1)  # (height, width, channels)
    num_classes: int = 1               # Binary: sigmoid output
    dropout_rate: float = 0.4
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 30
    early_stopping_patience: int = 7
    model_name: str = "predictive_maintenance_cnn"

    class Config:
        env_prefix = "MODEL_"


class APIConfig(BaseSettings):
    """FastAPI server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    max_upload_size_mb: int = 10
    allowed_audio_extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg")
    model_path: Path = Path("saved_models/predictive_maintenance_cnn.h5")

    class Config:
        env_prefix = "API_"


# ── Singleton instances used throughout the project ──────────────────────────
audio_cfg = AudioConfig()
data_cfg = DataConfig()
model_cfg = ModelConfig()
api_cfg = APIConfig()
