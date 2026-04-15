# PredictiveSense — Predictive Maintenance via Deep Learning on Audio

> **Production-grade** end-to-end system: synthetic machine audio → feature extraction → CNN classifier → FastAPI backend → browser UI.

---

## Architecture Overview

```
predictive_maintenance/
│
├── data/                    # Synthetic audio generation
│   ├── generate_data.py     ← Creates normal/ & faulty/ WAV files
│   ├── normal/              ← 500 healthy machine recordings
│   └── faulty/              ← 500 degraded machine recordings
│
├── preprocessing/
│   ├── audio_loader.py      ← WAV/MP3/FLAC loader + resampling
│   └── feature_extractor.py ← MFCC + Mel spectrogram → CNN tensor
│
├── models/
│   └── cnn_model.py         ← CNN architecture (build + load)
│
├── training/
│   ├── data_pipeline.py     ← Dataset builder + train/val/test split
│   └── train.py             ← Full training loop + evaluation + plots
│
├── inference/
│   └── predictor.py         ← Production inference service (singleton)
│
├── api/
│   ├── main.py              ← FastAPI app + endpoints
│   ├── schemas.py           ← Pydantic request/response models
│   └── middleware.py        ← Request logging middleware
│
├── frontend/
│   ├── index.html           ← Standalone browser UI (industrial dark theme)
│   └── streamlit_app.py     ← Richer Streamlit alternative
│
├── utils/
│   ├── logger.py            ← Loguru logging (console + rotating file)
│   └── helpers.py           ← timeit, normalize, label decoder, etc.
│
├── config/
│   └── settings.py          ← All config via Pydantic Settings
│
├── scripts/
│   ├── setup_and_run.sh     ← One-shot bootstrap script
│   └── demo_predict.py      ← Offline demo (no server needed)
│
├── tests/
│   ├── test_pipeline.py     ← Unit tests (data gen, loader, extractor)
│   └── test_api.py          ← API integration tests
│
├── saved_models/            ← Best checkpoint saved here after training
├── logs/                    ← Training curves, confusion matrix, metrics.json
├── requirements.txt
└── run_server.py            ← uvicorn launcher
```

---

## Quick Start (5 commands)

```bash
# 1. Clone / enter project
cd predictive_maintenance

# 2. Create virtualenv & install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Generate 1 000 synthetic WAV files (≈30 seconds)
python data/generate_data.py

# 4. Train the CNN (≈5–15 min on CPU, <2 min on GPU)
python training/train.py

# 5. Launch the API
python run_server.py
```

Then open:
- **API docs** → http://localhost:8000/docs
- **Browser UI** → http://localhost:8000/ui
- **Streamlit UI** → `streamlit run frontend/streamlit_app.py`

---

## Step-by-Step Walkthrough

### Step 1 — Synthetic Data Generation (`data/generate_data.py`)

Two sound classes are synthesised:

| Class  | Signal Construction |
|--------|---------------------|
| Normal | Harmonic stack (fundamental 50–120 Hz, 6 harmonics, decaying amplitudes) + tiny noise floor + smooth amplitude envelope |
| Faulty | Same base + **impulse spikes** (bearing damage) + **band-limited noise bursts** (electrical interference) + **FM drift** (shaft misalignment) + **amplitude jitter** (load fluctuations) |

All files are 3-second, 16-bit PCM WAV at 22 050 Hz.

### Step 2 — Feature Extraction (`preprocessing/feature_extractor.py`)

**Why MFCC?**
Mel-Frequency Cepstral Coefficients are the acoustic "fingerprint" of a sound:
1. STFT → power spectrum
2. Map to mel scale (perceptual frequency warping, sensitive to machine harmonics)
3. Log compression (matches ear's dynamic range)
4. DCT decorrelation → compact, uncorrelated coefficients

**Why Mel Spectrogram + CNN?**
A spectrogram is a 2-D image. CNNs detect 2-D patterns: horizontal bands = harmonics, vertical transients = impulse spikes, diagonal smearing = FM drift. Exactly the fault signatures we need.

Each clip → log-mel spectrogram → bicubic resize to 128×128 → normalise to [0,1] → (128,128,1) tensor.

### Step 3 — CNN Architecture (`models/cnn_model.py`)

```
Input (128, 128, 1)
    │
    ├─ Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout
    ├─ Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout
    ├─ Conv2D(128)→ BN → ReLU → Conv2D(128)→ BN → ReLU → MaxPool → Dropout
    ├─ Conv2D(256)→ BN → ReLU
    ├─ GlobalAveragePooling2D
    ├─ Dense(128) → ReLU → Dropout
    └─ Dense(1) → Sigmoid
            │
        P(Faulty) ∈ (0, 1)
```

- **Loss**: Binary cross-entropy
- **Optimiser**: Adam (lr=0.001)
- **Metrics**: Accuracy, Precision, Recall, AUC

### Step 4 — Training (`training/train.py`)

Callbacks:
- `EarlyStopping` — stops when `val_loss` stagnates (patience=7)
- `ReduceLROnPlateau` — halves LR on plateau (patience=3, min=1e-6)
- `ModelCheckpoint` — saves best weights by `val_auc`

Outputs in `logs/`:
- `training_curves.png` — loss & accuracy over epochs
- `confusion_matrix.png` — test-set classification matrix
- `metrics.json` — all numeric metrics

### Step 5 — Inference (`inference/predictor.py`)

The `Predictor` class is a **singleton service**:
- Model loaded once at API startup
- `predict(path)` — file path → result dict in <200ms
- `predict_bytes(bytes)` — for HTTP uploads (writes temp file)
- `predict_batch(paths)` — efficient batched forward pass

### Step 6 — API (`api/main.py`)

| Method | Endpoint        | Description                    |
|--------|-----------------|--------------------------------|
| GET    | `/health`       | Liveness probe                 |
| GET    | `/model/info`   | Model metadata                 |
| POST   | `/predict`      | Single file classification     |
| POST   | `/predict/batch`| Multi-file batch classification|
| GET    | `/ui`           | Serves the HTML frontend       |
| GET    | `/docs`         | Swagger UI                     |

Example response:
```json
{
  "request_id": "a1b2c3d4",
  "file": "motor_001.wav",
  "prediction": "Faulty",
  "confidence": 0.9732,
  "raw_probability": 0.9732
}
```

---

## Configuration

All settings are in `config/settings.py` and can be overridden via environment variables:

```bash
export AUDIO_SAMPLE_RATE=22050
export MODEL_EPOCHS=50
export API_PORT=8080
python run_server.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Expected Training Results

On 500+500 synthetic samples with the default architecture:

| Metric    | Typical Value |
|-----------|---------------|
| Test Acc  | 95–99%        |
| Test AUC  | 0.98–1.00     |
| Precision | 0.95–0.99     |
| Recall    | 0.95–0.99     |

*(Synthetic data is designed with clear separation; real-world data will score lower.)*

---

## Production Checklist

- [x] Modular architecture — one responsibility per file
- [x] All config centralised in `config/settings.py`
- [x] Structured JSON logging via loguru
- [x] Pydantic schema validation on all API I/O
- [x] CORS enabled (tighten `allow_origins` for production)
- [x] File size + extension validation before inference
- [x] EarlyStopping + LR scheduling prevents overfitting
- [x] Model checkpoint saves best weights automatically
- [x] Unit + integration test suite (pytest)
- [x] Reproducible via fixed random seed
- [ ] Containerise with Docker (`Dockerfile` — extend as needed)
- [ ] Add authentication (API key / OAuth2) before public deployment
- [ ] Replace in-memory dataset with `tf.data` pipeline for large datasets

---

## License

MIT — free to use, extend, and deploy.
