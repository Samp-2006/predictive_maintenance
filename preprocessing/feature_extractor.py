# ============================================================
# preprocessing/feature_extractor.py
# MFCC and Mel-spectrogram extraction.
#
# WHY MFCC?
# ----------
# Mel-Frequency Cepstral Coefficients compress the spectral shape
# of a signal into ~13–40 numbers per frame by:
#   1. Taking the STFT power spectrum
#   2. Mapping it onto the mel scale (perceptual frequency warping)
#   3. Applying log compression (mimics ear sensitivity)
#   4. Decorrelating with a Discrete Cosine Transform (DCT)
# MFCCs are compact, robust to minor timing shifts, and have been the
# gold standard in acoustic machine inspection for 30+ years.
#
# WHY MEL SPECTROGRAM WITH CNN?
# ------------------------------
# A mel spectrogram is a 2-D image (time × frequency).
# CNNs excel at 2-D pattern recognition — the same filters that
# detect edges and textures in photos detect harmonic structures
# and transients in spectrograms.  Unlike MFCCs, the spectrogram
# retains full spectral detail, letting the network decide which
# frequencies are discriminative.
# ============================================================

from typing import Tuple

import numpy as np
import librosa
import cv2       # used only for resizing spectrograms to fixed spatial size

from config.settings import audio_cfg, model_cfg
from utils.logger import logger
from utils.helpers import normalize_array


class FeatureExtractor:
    """
    Converts raw waveforms into ML-ready feature tensors.

    Outputs
    -------
    mel_spectrogram : np.ndarray  shape (n_mels, time_frames)
                      Log-power mel spectrogram — used as CNN input.
    mfcc            : np.ndarray  shape (n_mfcc, time_frames)
                      Cepstral coefficients — useful for statistics / audit.
    """

    def __init__(
        self,
        sr: int = audio_cfg.sample_rate,
        n_fft: int = audio_cfg.n_fft,
        hop_length: int = audio_cfg.hop_length,
        n_mels: int = audio_cfg.n_mels,
        n_mfcc: int = audio_cfg.n_mfcc,
        fmin: float = audio_cfg.fmin,
        fmax: float = audio_cfg.fmax,
        target_size: Tuple[int, int] = (128, 128),  # (H, W) fed to CNN
    ) -> None:
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax
        self.target_size = target_size  # resize so all inputs are identical

    # ── Core extraction ──────────────────────────────────────────────────

    def mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute log-power mel spectrogram.

        Pipeline:
            waveform  →  STFT  →  mel filterbank  →  log10  →  normalise

        Returns:
            2-D array of shape (n_mels, time_frames), values in [0, 1].
        """
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,   # power spectrum (magnitude²)
        )
        # Convert to decibels and normalise to [0, 1]
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return normalize_array(log_mel)

    def mfcc(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute MFCC matrix.

        Returns:
            2-D array of shape (n_mfcc, time_frames).
        """
        coeffs = librosa.feature.mfcc(
            y=waveform,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        return coeffs.astype(np.float32)

    def mfcc_with_deltas(self, waveform: np.ndarray) -> np.ndarray:
        """
        Stack MFCC + Δ + ΔΔ along channel axis.
        Delta features encode temporal dynamics (velocity + acceleration
        of spectral change), improving accuracy on transient faults.

        Returns:
            3-D array of shape (n_mfcc, time_frames, 3).
        """
        coeffs = self.mfcc(waveform)
        delta1 = librosa.feature.delta(coeffs, order=1)
        delta2 = librosa.feature.delta(coeffs, order=2)
        return np.stack([coeffs, delta1, delta2], axis=-1)  # (n_mfcc, T, 3)

    # ── CNN-ready tensor ─────────────────────────────────────────────────

    def extract_cnn_input(self, waveform: np.ndarray) -> np.ndarray:
        """
        Full pipeline: waveform → resized log-mel → (H, W, 1) float32 tensor.

        The channel dimension is added so TensorFlow/Keras Conv2D layers
        receive a standard image-like input.

        Returns:
            np.ndarray of shape (target_height, target_width, 1) in [0, 1].
        """
        spec = self.mel_spectrogram(waveform)          # (n_mels, T)

        # Resize to fixed spatial dimensions using bilinear interpolation
        # so every audio clip — regardless of length — maps to the same
        # tensor shape that the CNN expects.
        resized = cv2.resize(
            spec,
            (self.target_size[1], self.target_size[0]),   # cv2 takes (W, H)
            interpolation=cv2.INTER_LINEAR,
        )

        return resized[..., np.newaxis].astype(np.float32)  # add channel dim

    # ── Statistics helper (for debugging / EDA) ─────────────────────────

    def mfcc_statistics(self, waveform: np.ndarray) -> np.ndarray:
        """
        Flatten MFCC matrix to a 1-D feature vector of statistics:
        [mean, std, min, max] per coefficient → shape (n_mfcc × 4,).

        Useful for quick classical-ML experiments or sanity checks.
        """
        coeffs = self.mfcc(waveform)                          # (n_mfcc, T)
        stats = np.concatenate([
            coeffs.mean(axis=1),
            coeffs.std(axis=1),
            coeffs.min(axis=1),
            coeffs.max(axis=1),
        ])
        return stats.astype(np.float32)
