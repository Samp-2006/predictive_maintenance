# ============================================================
# utils/helpers.py
# Shared utility functions reused across the project.
# ============================================================

import time
import hashlib
import functools
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
from utils.logger import logger


# ── Timing decorator ─────────────────────────────────────────────────────────

def timeit(func: Callable) -> Callable:
    """
    Decorator that logs the wall-clock execution time of any function.
    Usage:
        @timeit
        def my_function(): ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


# ── File utilities ───────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_md5(path: str | Path) -> str:
    """Return the MD5 hex-digest of a file — useful for cache-busting."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Array utilities ──────────────────────────────────────────────────────────

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Min-max normalise to [0, 1].
    Avoids division-by-zero when the array is constant.
    """
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def pad_or_truncate(arr: np.ndarray, target_length: int, axis: int = -1) -> np.ndarray:
    """
    Ensure an array has exactly `target_length` along `axis`.
    Truncates if longer; zero-pads on the right if shorter.
    """
    current = arr.shape[axis]
    if current == target_length:
        return arr
    if current > target_length:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, target_length)
        return arr[tuple(slices)]
    # Pad
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target_length - current)
    return np.pad(arr, pad_width, mode="constant")


# ── Prediction helpers ───────────────────────────────────────────────────────

def label_from_probability(prob: float, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Convert a raw sigmoid probability to a human-readable result dict.

    Returns:
        {
            "prediction": "Normal" | "Faulty",
            "confidence": float,   # probability of the predicted class
            "raw_probability": float  # raw sigmoid output (P(Faulty))
        }
    """
    is_faulty = prob >= threshold
    confidence = float(prob) if is_faulty else float(1.0 - prob)
    return {
        "prediction": "Faulty" if is_faulty else "Normal",
        "confidence": round(confidence, 4),
        "raw_probability": round(float(prob), 6),
    }
