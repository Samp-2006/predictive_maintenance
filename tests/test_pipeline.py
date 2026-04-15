# ============================================================
# tests/test_pipeline.py
# Unit + integration tests for the core pipeline.
# Run:  pytest tests/ -v
# ============================================================

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import audio_cfg, model_cfg
from data.generate_data import generate_normal_sample, generate_faulty_sample
from preprocessing.audio_loader import AudioLoader
from preprocessing.feature_extractor import FeatureExtractor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def normal_wav(rng):
    """Generate a normal audio clip and write it to a temp WAV file."""
    audio = generate_normal_sample(rng=rng)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, audio_cfg.sample_rate, subtype="PCM_16")
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def faulty_wav(rng):
    audio = generate_faulty_sample(rng=rng)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, audio_cfg.sample_rate, subtype="PCM_16")
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


# ── Data generation tests ─────────────────────────────────────────────────────

class TestDataGeneration:
    def test_normal_sample_shape(self, rng):
        audio = generate_normal_sample(rng=rng)
        expected_length = int(audio_cfg.sample_rate * audio_cfg.duration)
        assert audio.shape == (expected_length,), f"Expected ({expected_length},), got {audio.shape}"

    def test_faulty_sample_shape(self, rng):
        audio = generate_faulty_sample(rng=rng)
        expected_length = int(audio_cfg.sample_rate * audio_cfg.duration)
        assert audio.shape == (expected_length,)

    def test_normal_signal_clipped(self, rng):
        audio = generate_normal_sample(rng=rng)
        assert np.abs(audio).max() <= 1.0, "Normal audio exceeds [-1, 1]"

    def test_faulty_signal_clipped(self, rng):
        audio = generate_faulty_sample(rng=rng)
        assert np.abs(audio).max() <= 1.0, "Faulty audio exceeds [-1, 1]"

    def test_signals_are_different(self, rng):
        n = generate_normal_sample(rng=rng)
        f = generate_faulty_sample(rng=rng)
        assert not np.allclose(n, f, atol=0.05), "Normal and faulty signals are too similar"


# ── Audio loader tests ────────────────────────────────────────────────────────

class TestAudioLoader:
    def test_load_normal(self, normal_wav):
        loader = AudioLoader()
        wav, sr = loader.load(normal_wav)
        assert wav.dtype == np.float32
        assert sr == audio_cfg.sample_rate
        assert wav.shape == (int(audio_cfg.sample_rate * audio_cfg.duration),)

    def test_load_faulty(self, faulty_wav):
        loader = AudioLoader()
        wav, sr = loader.load(faulty_wav)
        assert wav.shape == (int(audio_cfg.sample_rate * audio_cfg.duration),)

    def test_missing_file_raises(self):
        loader = AudioLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/audio.wav")

    def test_load_batch(self, normal_wav, faulty_wav):
        loader = AudioLoader()
        arr, names = loader.load_batch([normal_wav, faulty_wav])
        assert arr.shape[0] == 2
        assert len(names) == 2


# ── Feature extractor tests ───────────────────────────────────────────────────

class TestFeatureExtractor:
    @pytest.fixture(autouse=True)
    def setup(self, rng):
        self.rng = rng
        self.extractor = FeatureExtractor(target_size=(128, 128))
        self.waveform = generate_normal_sample(rng=rng)

    def test_mel_spectrogram_shape(self):
        spec = self.extractor.mel_spectrogram(self.waveform)
        assert spec.ndim == 2
        assert spec.shape[0] == audio_cfg.n_mels

    def test_mel_spectrogram_normalised(self):
        spec = self.extractor.mel_spectrogram(self.waveform)
        assert spec.min() >= 0.0
        assert spec.max() <= 1.0 + 1e-6

    def test_mfcc_shape(self):
        coeffs = self.extractor.mfcc(self.waveform)
        assert coeffs.shape[0] == audio_cfg.n_mfcc

    def test_cnn_input_shape(self):
        feature = self.extractor.extract_cnn_input(self.waveform)
        assert feature.shape == (128, 128, 1), f"Unexpected CNN input shape: {feature.shape}"

    def test_cnn_input_dtype(self):
        feature = self.extractor.extract_cnn_input(self.waveform)
        assert feature.dtype == np.float32

    def test_mfcc_with_deltas_shape(self):
        stacked = self.extractor.mfcc_with_deltas(self.waveform)
        assert stacked.shape == (audio_cfg.n_mfcc, stacked.shape[1], 3)

    def test_mfcc_statistics_shape(self):
        stats = self.extractor.mfcc_statistics(self.waveform)
        assert stats.shape == (audio_cfg.n_mfcc * 4,)
