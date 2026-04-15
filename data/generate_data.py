# ============================================================
# data/generate_data.py
# Synthetic machine audio generation.
#
# NORMAL sounds  → steady sinusoidal tones + gentle harmonic overtones
#                  mimicking a healthy motor or pump running at constant RPM.
# FAULTY sounds  → same base signal corrupted with:
#                  • random amplitude spikes (bearing chipping)
#                  • band-limited noise bursts (electrical interference)
#                  • frequency modulation drift (shaft misalignment)
#                  • impulse transients (gear tooth damage)
#
# Each sample is saved as a 16-bit PCM WAV file at 22 050 Hz.
# ============================================================

import sys
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Allow running as a standalone script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import audio_cfg, data_cfg
from utils.logger import logger
from utils.helpers import ensure_dir


# ── Signal building blocks ───────────────────────────────────────────────────

def _sine(freq: float, t: np.ndarray, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Pure sine tone."""
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def _harmonic_stack(
    fundamental: float,
    t: np.ndarray,
    n_harmonics: int = 5,
    decay: float = 0.6,
) -> np.ndarray:
    """
    Sum of harmonics (fundamental × 1, 2, 3, …) with exponentially decaying amplitudes.
    Models the tonal character of rotating machinery.
    """
    signal = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        signal += (decay ** (k - 1)) * _sine(fundamental * k, t)
    return signal / n_harmonics  # normalise to [-1, 1] range


def _white_noise(length: int, scale: float = 0.01) -> np.ndarray:
    """Additive white Gaussian noise — always present at low level."""
    return np.random.normal(0, scale, length)


def _random_spikes(length: int, rate: float = 0.005, amplitude: float = 0.8) -> np.ndarray:
    """
    Impulse transients — sparse random spikes.
    Models gear tooth damage or bearing race defects.
    """
    spikes = np.zeros(length)
    n_spikes = max(1, int(length * rate))
    positions = np.random.randint(0, length, n_spikes)
    spikes[positions] = amplitude * np.random.choice([-1, 1], n_spikes)
    return spikes


def _noise_burst(length: int, center: float, bandwidth: float, amplitude: float) -> np.ndarray:
    """
    Band-limited noise burst centred at `center` Hz.
    Models electrical interference or cavitation.
    """
    noise = np.random.normal(0, 1, length)
    # Simple band-pass via FFT mask
    freqs = np.fft.rfftfreq(length, d=1 / audio_cfg.sample_rate)
    spectrum = np.fft.rfft(noise)
    mask = np.abs(freqs - center) < bandwidth / 2
    spectrum[~mask] = 0
    return amplitude * np.fft.irfft(spectrum, n=length)


def _fm_drift(signal: np.ndarray, t: np.ndarray, drift_rate: float = 2.0) -> np.ndarray:
    """
    Slow sinusoidal frequency modulation.
    Models shaft misalignment causing RPM fluctuation.
    """
    phase_mod = np.sin(2 * np.pi * drift_rate * t) * 0.3
    # Shift signal in time via phase rotation in frequency domain
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal))
    spectrum *= np.exp(1j * 2 * np.pi * freqs * phase_mod.mean() * len(signal))
    return np.fft.irfft(spectrum, n=len(signal))


# ── Sample generators ────────────────────────────────────────────────────────

def generate_normal_sample(
    sr: int = audio_cfg.sample_rate,
    duration: float = audio_cfg.duration,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Healthy machine signal:
    • Fundamental motor frequency (50–120 Hz) + harmonics
    • Very low white noise floor
    • No transients
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    fundamental = rng.uniform(50, 120)       # RPM-derived frequency
    signal = _harmonic_stack(fundamental, t, n_harmonics=6, decay=0.55)
    signal += _white_noise(len(t), scale=0.005)   # tiny noise floor

    # Soft amplitude envelope (machines don't start/stop abruptly)
    envelope = np.ones_like(t)
    fade = int(0.05 * sr)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    signal *= envelope

    return _normalise_signal(signal)


def generate_faulty_sample(
    sr: int = audio_cfg.sample_rate,
    duration: float = audio_cfg.duration,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Faulty machine signal — base tone degraded by multiple fault modes,
    each applied with randomised severity so no two faults are identical.

    Fault modes (applied stochastically):
        1. Bearing spikes   — impulse transients
        2. Noise burst      — band-limited interference
        3. FM drift         — shaft misalignment wobble
        4. Amplitude jitter — motor load fluctuations
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    fundamental = rng.uniform(50, 120)
    signal = _harmonic_stack(fundamental, t, n_harmonics=4, decay=0.5)

    # ① Bearing spikes (always present in faulty machines)
    signal += _random_spikes(len(t), rate=rng.uniform(0.003, 0.015),
                             amplitude=rng.uniform(0.4, 0.9))

    # ② Band-limited noise burst (50 % chance)
    if rng.random() > 0.5:
        centre = rng.uniform(200, 3000)
        signal += _noise_burst(len(t), centre, bandwidth=rng.uniform(50, 400),
                               amplitude=rng.uniform(0.1, 0.35))

    # ③ Frequency modulation drift (70 % chance)
    if rng.random() > 0.3:
        signal = _fm_drift(signal, t, drift_rate=rng.uniform(1.0, 5.0))

    # ④ Amplitude jitter (random gain modulation)
    jitter_freq = rng.uniform(0.5, 4.0)
    jitter = 1.0 + rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * jitter_freq * t)
    signal *= jitter

    # Background noise (higher than normal)
    signal += _white_noise(len(t), scale=rng.uniform(0.02, 0.08))

    return _normalise_signal(signal)


def _normalise_signal(signal: np.ndarray) -> np.ndarray:
    """Peak-normalise to ±0.95 (avoid digital clipping)."""
    peak = np.abs(signal).max()
    if peak < 1e-8:
        return signal
    return (signal / peak * 0.95).astype(np.float32)


# ── Dataset generation ───────────────────────────────────────────────────────

def generate_dataset(
    n_normal: int = data_cfg.n_normal_samples,
    n_faulty: int = data_cfg.n_faulty_samples,
    seed: int = data_cfg.random_seed,
) -> None:
    """
    Generate and save synthetic WAV files.

    Args:
        n_normal: Number of normal machine recordings to create.
        n_faulty: Number of faulty machine recordings to create.
        seed:     Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    normal_dir = ensure_dir(data_cfg.normal_dir)
    faulty_dir = ensure_dir(data_cfg.faulty_dir)

    logger.info(f"Generating {n_normal} normal samples → {normal_dir}")
    for i in tqdm(range(n_normal), desc="Normal", unit="file"):
        audio = generate_normal_sample(rng=rng)
        path = normal_dir / f"normal_{i:04d}.wav"
        sf.write(path, audio, audio_cfg.sample_rate, subtype="PCM_16")

    logger.info(f"Generating {n_faulty} faulty samples → {faulty_dir}")
    for i in tqdm(range(n_faulty), desc="Faulty", unit="file"):
        audio = generate_faulty_sample(rng=rng)
        path = faulty_dir / f"faulty_{i:04d}.wav"
        sf.write(path, audio, audio_cfg.sample_rate, subtype="PCM_16")

    logger.success(
        f"Dataset ready: {n_normal} normal + {n_faulty} faulty files "
        f"in '{data_cfg.data_dir}'"
    )


# ── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_dataset()
