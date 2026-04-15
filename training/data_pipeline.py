# ============================================================
# training/data_pipeline.py
# Loads audio files, extracts CNN features, and returns
# train / validation / test splits as NumPy arrays.
#
# Design decisions:
#   • All feature extraction happens here (not inside the model)
#     so the same pipeline is reused during inference.
#   • Arrays are pre-computed and held in memory.  For datasets
#     > RAM capacity, swap np.stack for a tf.data pipeline with
#     librosa called inside a Python generator.
# ============================================================

import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import audio_cfg, data_cfg, model_cfg
from preprocessing.audio_loader import AudioLoader
from preprocessing.feature_extractor import FeatureExtractor
from utils.logger import logger
from utils.helpers import ensure_dir


# ── Label constants ──────────────────────────────────────────────────────────
LABEL_NORMAL = 0
LABEL_FAULTY = 1


def _collect_paths(directory: Path, extensions: tuple = (".wav", ".mp3", ".flac")) -> List[Path]:
    """Recursively collect audio file paths from a directory."""
    paths = []
    for ext in extensions:
        paths.extend(sorted(directory.glob(f"**/*{ext}")))
    return paths


def build_dataset(
    normal_dir: Path = data_cfg.normal_dir,
    faulty_dir: Path = data_cfg.faulty_dir,
    target_size: Tuple[int, int] = (
        model_cfg.input_shape[0],
        model_cfg.input_shape[1],
    ),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all audio files, extract CNN-ready features, and return (X, y).

    Args:
        normal_dir:  Directory containing 'normal' WAV files.
        faulty_dir:  Directory containing 'faulty' WAV files.
        target_size: (H, W) spatial size each spectrogram is resized to.

    Returns:
        X : float32 array of shape (N, H, W, 1)
        y : float32 array of shape (N,) with values 0.0 or 1.0
    """
    loader    = AudioLoader(target_sr=audio_cfg.sample_rate, duration=audio_cfg.duration)
    extractor = FeatureExtractor(target_size=target_size)

    normal_paths = _collect_paths(Path(normal_dir))
    faulty_paths = _collect_paths(Path(faulty_dir))

    logger.info(
        f"Found {len(normal_paths)} normal + {len(faulty_paths)} faulty files"
    )

    X, y = [], []

    for paths, label, tag in [
        (normal_paths, LABEL_NORMAL, "Normal"),
        (faulty_paths, LABEL_FAULTY, "Faulty"),
    ]:
        for p in tqdm(paths, desc=f"Extracting {tag}", unit="file"):
            try:
                waveform, _ = loader.load(p)
                feature     = extractor.extract_cnn_input(waveform)   # (H, W, 1)
                X.append(feature)
                y.append(float(label))
            except Exception as exc:
                logger.warning(f"Skipping {p.name}: {exc}")

    if not X:
        raise RuntimeError("Feature matrix is empty — did you generate the dataset first?")

    X_arr = np.stack(X, axis=0).astype(np.float32)   # (N, H, W, 1)
    y_arr = np.array(y, dtype=np.float32)             # (N,)

    logger.info(f"Dataset shape: X={X_arr.shape}  y={y_arr.shape}")
    logger.info(
        f"Class balance — Normal: {(y_arr==0).sum()}  Faulty: {(y_arr==1).sum()}"
    )
    return X_arr, y_arr


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = data_cfg.train_ratio,
    val_ratio: float   = data_cfg.val_ratio,
    seed: int          = data_cfg.random_seed,
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Stratified train / validation / test split.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train + val ratios must sum to < 1.0"

    # First split: train vs (val + test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=(1.0 - train_ratio),
        stratify=y,
        random_state=seed,
    )

    # Second split: val vs test (relative to X_tmp size)
    relative_val = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(1.0 - relative_val),
        stratify=y_tmp,
        random_state=seed,
    )

    logger.info(
        f"Split — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}"
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
