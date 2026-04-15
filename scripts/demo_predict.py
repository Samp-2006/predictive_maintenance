# ============================================================
# scripts/demo_predict.py
# Quick offline demo: generate one sample of each class,
# run the predictor, and print results — no server needed.
#
# Usage:
#   python scripts/demo_predict.py
# ============================================================

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_data import generate_normal_sample, generate_faulty_sample
from config.settings import audio_cfg
from inference.predictor import Predictor
from utils.logger import logger


def main() -> None:
    predictor = Predictor()
    try:
        predictor.load()
    except FileNotFoundError:
        logger.error(
            "No trained model found.\n"
            "Run: python training/train.py  first."
        )
        sys.exit(1)

    rng = np.random.default_rng(777)

    for label, generator in [
        ("NORMAL", generate_normal_sample),
        ("FAULTY", generate_faulty_sample),
    ]:
        audio = generator(rng=rng)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, audio_cfg.sample_rate, subtype="PCM_16")
            result = predictor.predict(tmp.name)
            Path(tmp.name).unlink(missing_ok=True)

        correct = result["prediction"].upper() == label
        status  = "✅ CORRECT" if correct else "❌ WRONG"
        print(
            f"\nGround truth : {label}\n"
            f"Prediction   : {result['prediction']}\n"
            f"Confidence   : {result['confidence']*100:.1f}%\n"
            f"P(Faulty)    : {result['raw_probability']:.4f}\n"
            f"Status       : {status}\n"
            + "─" * 40
        )


if __name__ == "__main__":
    main()
