# ============================================================
# frontend/streamlit_app.py
# Alternative Streamlit UI — richer visuals, zero JS.
#
# Run:
#   streamlit run frontend/streamlit_app.py
# ============================================================

import sys
import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
import librosa
import librosa.display

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictiveSense",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme injection ──────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #080c10; }
  [data-testid="stSidebar"]          { background: #0d1318; }
  .stButton button {
    background: transparent;
    border: 1px solid #007a99;
    color: #00e5ff;
    font-weight: bold;
    letter-spacing: 3px;
    text-transform: uppercase;
    width: 100%;
  }
  .stButton button:hover { background: #00e5ff; color: #080c10; }
  .verdict-box {
    padding: 24px;
    border-radius: 4px;
    text-align: center;
    font-size: 2em;
    font-weight: 900;
    letter-spacing: 4px;
  }
  .normal { background: rgba(0,255,136,0.08); border: 1px solid #00804a; color: #00ff88; }
  .faulty { background: rgba(255,62,62,0.08); border: 1px solid #801f1f; color: #ff3e3e; }
</style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8000"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ System")
    api_url = st.text_input("API URL", value=API_BASE)

    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        h = r.json()
        if h["model_loaded"]:
            st.success("✅ API Online · Model Ready")
        else:
            st.warning("⚠ API Online · Model Not Loaded")
    except Exception:
        st.error("❌ API Offline")

    st.divider()
    st.markdown("### About")
    st.markdown("""
    **PredictiveSense** uses a CNN trained on mel spectrograms of machine audio
    to classify equipment health as **Normal** or **Faulty**.

    **Fault modes detected:**
    - Bearing impulse spikes
    - Band-limited electrical noise
    - Frequency modulation drift
    - Amplitude jitter (load fluctuation)
    """)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🎙 PredictiveSense")
st.markdown("##### Industrial Machine Health Monitor — Deep Learning Audio Classifier")
st.divider()

uploaded = st.file_uploader(
    "Upload machine audio clip",
    type=["wav", "mp3", "flac", "ogg"],
    help="Drag & drop or click to browse. Max 10 MB.",
)

if uploaded:
    audio_bytes = uploaded.read()

    col1, col2 = st.columns([1, 1])

    # ── Waveform & Spectrogram ────────────────────────────────────────────
    with col1:
        st.markdown("#### Waveform")
        st.audio(audio_bytes, format="audio/wav")

        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True, duration=3.0)

            fig, ax = plt.subplots(figsize=(6, 2), facecolor="#0d1318")
            ax.set_facecolor("#0d1318")
            times = np.linspace(0, len(y) / sr, len(y))
            ax.plot(times, y, color="#007a99", linewidth=0.6)
            ax.set_xlabel("Time (s)", color="#4a6070", fontsize=9)
            ax.set_ylabel("Amplitude", color="#4a6070", fontsize=9)
            ax.tick_params(colors="#4a6070")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2d3d")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as exc:
            st.warning(f"Could not render waveform: {exc}")

    with col2:
        st.markdown("#### Mel Spectrogram")
        try:
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_mel = librosa.power_to_db(mel, ref=np.max)

            fig2, ax2 = plt.subplots(figsize=(6, 2.5), facecolor="#0d1318")
            ax2.set_facecolor("#0d1318")
            img = librosa.display.specshow(
                log_mel, sr=sr, x_axis="time", y_axis="mel",
                ax=ax2, cmap="magma",
            )
            fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
            ax2.set_title("Log Mel Spectrogram", color="#4a6070", fontsize=9)
            ax2.tick_params(colors="#4a6070")
            for spine in ax2.spines.values():
                spine.set_edgecolor("#1e2d3d")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
        except Exception as exc:
            st.warning(f"Could not render spectrogram: {exc}")

    st.divider()

    # ── Predict button ────────────────────────────────────────────────────
    if st.button("🔍  ANALYSE SIGNAL", use_container_width=True):
        with st.spinner("Running inference…"):
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    f"{api_url}/predict",
                    files={"file": (uploaded.name, audio_bytes, "audio/wav")},
                    timeout=30,
                )
                latency_ms = round((time.perf_counter() - t0) * 1000)

                if resp.ok:
                    data = resp.json()
                    is_faulty = data["prediction"] == "Faulty"
                    cls = "faulty" if is_faulty else "normal"
                    icon = "🔴" if is_faulty else "🟢"

                    st.markdown(
                        f'<div class="verdict-box {cls}">'
                        f'{icon} &nbsp; {data["prediction"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Confidence",   f"{round(data['confidence']*100, 1)}%")
                    c2.metric("P(Faulty)",    f"{data['raw_probability']:.4f}")
                    c3.metric("Latency",      f"{latency_ms} ms")

                    st.markdown(
                        "⚠️ **FAULT DETECTED — Schedule maintenance immediately.**"
                        if is_faulty else
                        "✅ **Machine operating within normal parameters.**"
                    )
                else:
                    st.error(f"API error {resp.status_code}: {resp.text[:200]}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Is the FastAPI server running on port 8000?")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

else:
    st.info("👆 Upload a WAV / MP3 / FLAC / OGG audio file to begin analysis.")
