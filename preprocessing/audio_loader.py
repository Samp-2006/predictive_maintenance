# ============================================================
# preprocessing/audio_loader.py
# Robust audio file loading with resampling and duration control.
#
# Why separate this?  Loading is distinct from feature extraction.
# We want a single place that handles all file-format quirks so
# every downstream module receives a consistent float32 waveform.
# ============================================================

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import librosa

from config.settings import audio_cfg
from utils.logger import logger
from utils.helpers import pad_or_truncate


class AudioLoader:
    """
    Loads an audio file and returns a mono float32 NumPy array
    resampled to `target_sr` and cropped / padded to `duration` seconds.

    Supported formats: WAV, MP3, FLAC, OGG (anything librosa can read).
    """

    def __init__(
        self,
        target_sr: int = audio_cfg.sample_rate,
        duration: float = audio_cfg.duration,
    ) -> None:
        self.target_sr = target_sr
        self.duration = duration
        self.n_samples = int(target_sr * duration)

    # ── Public API ───────────────────────────────────────────────────────

    def load(self, path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Load an audio file.

        Returns:
            (waveform, sample_rate) where waveform.shape == (n_samples,)

        Raises:
            FileNotFoundError: If the path does not exist.
            RuntimeError:      If librosa cannot decode the file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            waveform, sr = librosa.load(
                str(path),
                sr=self.target_sr,   # librosa resamples automatically
                mono=True,           # always downmix to mono
                duration=self.duration,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio '{path}': {exc}") from exc

        # Guarantee exact length (librosa may return slightly fewer samples
        # for some compressed formats)
        waveform = pad_or_truncate(waveform, self.n_samples)

        logger.debug(f"Loaded '{path.name}' — shape={waveform.shape}, sr={sr}")
        return waveform.astype(np.float32), sr

    def load_batch(
        self,
        paths: list[str | Path],
        verbose: bool = False,
    ) -> Tuple[np.ndarray, list[str]]:
        """
        Load multiple files into a stacked array.

        Returns:
            (array of shape [N, n_samples], list of filenames)
        """
        waveforms, names = [], []
        for p in paths:
            try:
                wav, _ = self.load(p)
                waveforms.append(wav)
                names.append(Path(p).name)
            except Exception as exc:
                logger.warning(f"Skipping '{p}': {exc}")

        if not waveforms:
            raise RuntimeError("No audio files could be loaded.")

        return np.stack(waveforms, axis=0), names
