# ═══════════════════════════════════════════════════════════
# app.py | Deepfake Audio Detection
# MARS Club Open Project 2026 — IIT Roorkee
# Author: Sameer Modi (23410030)
#
# Selected model: LCNN (MFM activation) — ALL 5 CRITERIA PASS
#   Accuracy : 88.51%  ✓ (target ≥ 80%)
#   EER      : 11.58%  ✓ (target ≤ 12%)
#   F1 Score : 88.52%  ✓ (target ≥ 80%)
#   Real acc : 88.42%  ✓ (target ≥ 75%)
#   Fake acc : 88.60%  ✓ (target ≥ 75%)
#
# Pipeline matches notebook exactly:
#   LFCC hyperparameters  → Cell 4
#   Normalization         → Cell 7
#   LCNN architecture     → Cell 10
#   MFM activation        → Cell 8
#   Threshold logic       → Cell 11
# ═══════════════════════════════════════════════════════════

import os
import pickle
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
from scipy.fftpack import dct

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Deepfake Audio Detector",
    page_icon  = "🎙️",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────
# LFCC HYPERPARAMETERS — must match notebook Cell 4 exactly
# ─────────────────────────────────────────────────────────────
TARGET_SR    = 16000
N_LFCC       = 40
N_FILTER     = 70
N_FFT        = 512
HOP_LENGTH   = 160
FIXED_FRAMES = 94

# ─────────────────────────────────────────────────────────────
# MFM ACTIVATION — must be defined before load_assets()
# Same function as notebook Cell 8
# Required as custom_object when loading LCNN from .h5
# ─────────────────────────────────────────────────────────────
def mfm_activation(x):
    """
    Max Feature Map (MFM) activation — from notebook Cell 8.
    Splits input along last axis into two equal halves.
    Returns element-wise maximum — competitive activation.
    Input  shape: (..., 2N)
    Output shape: (..., N)
    MUST be passed as custom_objects when loading LCNN.
    """
    import tensorflow as tf
    n  = tf.shape(x)[-1] // 2
    x1 = x[..., :n]
    x2 = x[..., n:]
    return tf.maximum(x1, x2)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL — cached, runs once per session
# LCNN selected as winner (ALL 5 criteria pass)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    """
    Load LCNN model, normalization params, and threshold.
    LCNN requires custom_objects={'mfm_activation': mfm_activation}
    because it uses Lambda layers wrapping mfm_activation.
    """
    with open("norm_params.pkl", "rb") as f:
        norm = pickle.load(f)
    with open("threshold.pkl", "rb") as f:
        thresh = pickle.load(f)

    mean      = norm["mean"]
    std       = norm["std"]
    threshold = thresh["optimal_threshold"]

    # Method 1: Load LCNN directly with custom_objects
    try:
        from tensorflow.keras.models import load_model
        model = load_model(
            "best_model.h5",
            custom_objects={"mfm_activation": mfm_activation}
        )
        return model, mean, std, threshold
    except Exception:
        pass

    # Method 2: Rebuild LCNN architecture + load weights
    # (fallback if direct load fails due to version mismatch)
    import tensorflow as tf
    from tensorflow.keras.layers import (
        BatchNormalization, Conv2D, Dense,
        Dropout, Flatten, Input, Lambda, MaxPooling2D,
    )
    from tensorflow.keras.models import Model

    inputs = Input(shape=(40, 94, 1))
    # Block 1: Conv(64)+MFM → 32ch → Pool → (20,47,32)
    x = Conv2D(64, (3,3), padding="same", use_bias=False)(inputs)
    x = Lambda(mfm_activation, name="mfm_1")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x)
    # Block 2: Conv(128)+MFM → 64ch → Pool → (10,23,64)
    x = Conv2D(128, (3,3), padding="same", use_bias=False)(x)
    x = Lambda(mfm_activation, name="mfm_2")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x)
    # Block 3: Conv(256)+MFM → 128ch → Pool → (5,11,128)
    x = Conv2D(256, (3,3), padding="same", use_bias=False)(x)
    x = Lambda(mfm_activation, name="mfm_3")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x)
    # Dense head: Dense(256)+MFM → 128 units
    x = Flatten()(x)
    x = Dense(256, use_bias=False)(x)
    x = Lambda(mfm_activation, name="mfm_dense")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="LCNN_MFM")
    model.load_weights("best_model.h5")
    return model, mean, std, threshold


# ─────────────────────────────────────────────────────────────
# LFCC EXTRACTION — identical to notebook Cell 4
# ─────────────────────────────────────────────────────────────
def extract_lfcc(audio: np.ndarray) -> np.ndarray:
    """
    Extract LFCC features from raw 1D mono audio.
    Matches notebook Cell 4 exactly.
    Input : float32 (N,)
    Output: float32 (40, 94)
    """
    # Step 1: STFT power spectrogram
    stft = np.abs(
        np.fft.rfft(
            np.array([
                audio[i:i+N_FFT] if i+N_FFT <= len(audio)
                else np.pad(audio[i:], (0, N_FFT-len(audio[i:])))
                for i in range(0, len(audio)-N_FFT+HOP_LENGTH, HOP_LENGTH)
            ]),
            n=N_FFT,
        )
    ) ** 2
    stft = stft.T  # (257, T)

    # Step 2: Linear filterbank
    freqs        = np.linspace(0, TARGET_SR/2, N_FFT//2+1)
    filter_edges = np.linspace(0, TARGET_SR/2, N_FILTER+2)
    filterbank   = np.zeros((N_FILTER, N_FFT//2+1))
    for i in range(N_FILTER):
        l, c, r = filter_edges[i], filter_edges[i+1], filter_edges[i+2]
        for j, f in enumerate(freqs):
            if l <= f <= c:   filterbank[i,j] = (f-l)/(c-l+1e-8)
            elif c < f <= r:  filterbank[i,j] = (r-f)/(r-c+1e-8)

    # Step 3: Log filter energies
    log_e = np.log(np.dot(filterbank, stft) + 1e-8)

    # Step 4: DCT — keep top 40
    lfcc = dct(log_e, type=2, axis=0, norm="ortho")[:N_LFCC]

    # Step 5: Pad or truncate
    T = lfcc.shape[1]
    if T < FIXED_FRAMES:
        lfcc = np.pad(lfcc, ((0,0),(0,FIXED_FRAMES-T)), mode="constant")
    else:
        lfcc = lfcc[:, :FIXED_FRAMES]
    return lfcc  # (40, 94)


def run_prediction(audio, model, mean, std, threshold):
    """
    Full prediction pipeline matching notebook:
    Cell 7 normalization + Cell 11 threshold logic.
    """
    lfcc      = extract_lfcc(audio)
    lfcc_norm = (lfcc - mean) / (std + 1e-8)
    X         = lfcc_norm[np.newaxis, ..., np.newaxis]  # (1,40,94,1)
    prob      = float(model.predict(X, verbose=0)[0][0])
    label     = "FAKE" if prob > threshold else "REAL"
    conf      = prob * 100 if prob > threshold else (1 - prob) * 100
    return label, conf, prob, lfcc


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ Deepfake Detector")
    st.caption("MARS Club · IIT Roorkee · 2026")
    st.divider()

    st.markdown("### 🏆 Final Results")
    st.success("**ALL 5 criteria: PASS**")
    st.markdown("**Selected model: LCNN (MFM)**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy",  "88.51%", delta="+8.51%")
        st.metric("F1 Score",  "88.52%", delta="+8.52%")
        st.metric("EER ↓",     "11.58%")
    with col2:
        st.metric("Real Acc",  "88.42%", delta="+13.42%")
        st.metric("Fake Acc",  "88.60%", delta="+13.60%")

    st.divider()
    st.markdown("### 📊 CNN vs LCNN")
    st.markdown("""
| Metric | CNN | LCNN ✓ |
|---|---|---|
| Accuracy | 86.49 | **88.51** |
| F1 Score | 86.27 | **88.52** |
| Real Acc | 88.05 | **88.42** |
| Fake Acc | 84.93 | **88.60** |
| EER ↓ | 13.24 | **11.58** |
""")
    st.caption("LCNN wins 5/5 → deployed ✓")

    st.divider()
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
| | |
|---|---|
| Model | LCNN + MFM activation |
| Features | LFCC 40×94 |
| Augmentation | Noise+Stretch+Gain |
| Train samples | ~55,824 |
| Dataset | FoR for-2sec |
| Test samples | 1,088 |
""")
    st.divider()
    st.caption("Sameer Modi · 23410030")
    st.caption("Geological Technology · IIT Roorkee")


# ─────────────────────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading LCNN model..."):
    try:
        model, MEAN, STD, THRESHOLD = load_assets()
        model_ok = True
    except Exception as e:
        model_ok = False
        load_err = str(e)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("🎙️ Deepfake Audio Detector")
st.markdown(
    "Upload a `.wav` speech file · Get an instant **REAL / FAKE** verdict · "
    "LCNN with MFM activation · ALL 5 MARS Club criteria passed · IIT Roorkee 2026"
)
st.divider()

if not model_ok:
    st.error(f"❌ Model load failed: {load_err}")
    st.info(
        "Make sure **best_model.h5**, **norm_params.pkl**, and **threshold.pkl** "
        "are in the same folder as app.py"
    )
    st.stop()

st.success(
    f"✅ LCNN model loaded — threshold {THRESHOLD:.3f} — "
    f"Accuracy 88.51% — EER 11.58% — ALL 5 criteria PASS"
)


# ─────────────────────────────────────────────────────────────
# UPLOAD + PREDICTION
# ─────────────────────────────────────────────────────────────
st.subheader("📤 Upload Audio")

uploaded = st.file_uploader(
    "Drop a WAV file here or click Browse",
    type=["wav"],
    key="wav_upload",
    help="WAV format · 2–5 seconds of clear speech · single speaker",
)

if uploaded is not None:

    col_play, col_info = st.columns([2, 1])
    with col_play:
        st.markdown("**▶ Playback**")
        st.audio(uploaded)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    try:
        audio, sr = sf.read(tmp_path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio    = audio.astype(np.float32)
        duration = len(audio) / sr

        with col_info:
            st.markdown("**File info**")
            st.markdown(f"- Duration: **{duration:.2f}s**")
            st.markdown(f"- Sample rate: **{sr:,} Hz**")
            st.markdown(f"- Samples: **{len(audio):,}**")

        st.divider()

        # Prediction
        with st.spinner("🔬 Analysing LFCC fingerprint..."):
            label, conf, prob, lfcc = run_prediction(
                audio, model, MEAN, STD, THRESHOLD
            )

        # VERDICT
        if label == "REAL":
            st.success("## ✅ GENUINE HUMAN SPEECH")
            accent = "#1D9E75"
        else:
            st.error("## 🚨 AI-GENERATED DEEPFAKE")
            accent = "#D85A30"

        # METRICS
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Verdict",    label)
        m2.metric("Confidence", f"{conf:.1f}%")
        m3.metric("Fake score", f"{prob*100:.2f}%")
        m4.metric("Real score", f"{(1-prob)*100:.2f}%")
        m5.metric("Threshold",  f"{THRESHOLD:.3f}")

        # PROBABILITY BAR
        st.markdown("**Probability split — Real vs Fake**")
        st.progress(
            float(1 - prob),
            text=f"Real {(1-prob)*100:.1f}%  ←→  Fake {prob*100:.1f}%"
        )

        # INTERPRETATION
        if label == "FAKE":
            st.warning(
                f"⚠️ **Deepfake detected.** "
                f"Raw score {prob:.4f} exceeds threshold {THRESHOLD:.3f}. "
                f"The LFCC fingerprint shows high-frequency artefacts "
                f"characteristic of TTS/vocoder-generated speech."
            )
        else:
            st.info(
                f"ℹ️ **Genuine speech confirmed.** "
                f"Raw score {prob:.4f} is below threshold {THRESHOLD:.3f}. "
                f"Natural frequency patterns consistent with human vocal tract."
            )

        st.divider()

        # ANALYSIS TABS
        st.subheader("🔬 Audio Analysis")
        tab1, tab2, tab3 = st.tabs([
            "📊 Fingerprint & Waveform",
            "📈 LFCC Statistics",
            "ℹ️ How It Works",
        ])

        with tab1:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            fig.patch.set_facecolor("#0E1017")

            # LFCC heatmap
            im = axes[0].imshow(
                lfcc, aspect="auto", origin="lower", cmap="plasma"
            )
            axes[0].set_facecolor("#0E1017")
            axes[0].set_title(
                f"LFCC Fingerprint — {label}",
                color="#E8EAF0", fontsize=10, fontweight="bold", pad=8
            )
            axes[0].set_xlabel("Time frames (94)", color="#8B92A5", fontsize=8)
            axes[0].set_ylabel("Coefficients (40)", color="#8B92A5", fontsize=8)
            axes[0].tick_params(colors="#8B92A5", labelsize=7)
            for sp in axes[0].spines.values():
                sp.set_edgecolor("#1E2130")
            cb = plt.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)
            cb.ax.tick_params(colors="#8B92A5", labelsize=7)

            # Waveform
            t = np.linspace(0, duration, len(audio))
            axes[1].plot(t, audio, color=accent, linewidth=0.7, alpha=0.9)
            axes[1].fill_between(t, audio, 0, color=accent, alpha=0.08)
            axes[1].axhline(0, color="#1E2130", linewidth=0.5)
            axes[1].set_facecolor("#0E1017")
            axes[1].set_title(
                "Waveform", color="#E8EAF0",
                fontsize=10, fontweight="bold", pad=8
            )
            axes[1].set_xlabel("Time (s)", color="#8B92A5", fontsize=8)
            axes[1].set_ylabel("Amplitude", color="#8B92A5", fontsize=8)
            axes[1].tick_params(colors="#8B92A5", labelsize=7)
            axes[1].grid(True, alpha=0.06)
            for sp in axes[1].spines.values():
                sp.set_edgecolor("#1E2130")

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Duration",    f"{duration:.2f}s")
            c2.metric("Sample rate", f"{sr:,} Hz")
            c3.metric("LFCC shape",  "40 × 94")
            c4.metric("Model",       "LCNN (MFM)")

        with tab2:
            ca, cb_col = st.columns(2)

            with ca:
                fig2, ax = plt.subplots(figsize=(6, 3.5))
                fig2.patch.set_facecolor("#0E1017")
                ax.set_facecolor("#0E1017")
                coeff_means = lfcc.mean(axis=1)
                coeff_stds  = lfcc.std(axis=1)
                x_c = np.arange(N_LFCC)
                ax.fill_between(x_c,
                                coeff_means - coeff_stds,
                                coeff_means + coeff_stds,
                                color=accent, alpha=0.15)
                ax.plot(x_c, coeff_means, color=accent, linewidth=1.5)
                ax.axhline(0, color="#1E2130", linewidth=0.5)
                ax.set_title("Mean LFCC per coefficient",
                             color="#E8EAF0", fontsize=9, pad=6)
                ax.set_xlabel("Coefficient index", color="#8B92A5", fontsize=8)
                ax.set_ylabel("Value", color="#8B92A5", fontsize=8)
                ax.tick_params(colors="#8B92A5", labelsize=7)
                ax.grid(True, alpha=0.05)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#1E2130")
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close()

            with cb_col:
                fig3, ax = plt.subplots(figsize=(6, 3.5))
                fig3.patch.set_facecolor("#0E1017")
                ax.set_facecolor("#0E1017")
                energy   = (lfcc ** 2).mean(axis=0)
                t_frames = np.arange(FIXED_FRAMES) * (HOP_LENGTH / TARGET_SR)
                ax.fill_between(t_frames, energy, color=accent, alpha=0.25)
                ax.plot(t_frames, energy, color=accent, linewidth=1.2)
                ax.set_title("Frame energy over time",
                             color="#E8EAF0", fontsize=9, pad=6)
                ax.set_xlabel("Time (s)", color="#8B92A5", fontsize=8)
                ax.set_ylabel("Mean sq. energy", color="#8B92A5", fontsize=8)
                ax.tick_params(colors="#8B92A5", labelsize=7)
                ax.grid(True, alpha=0.05)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#1E2130")
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close()

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("LFCC min",  f"{lfcc.min():.3f}")
            s2.metric("LFCC max",  f"{lfcc.max():.3f}")
            s3.metric("LFCC mean", f"{lfcc.mean():.3f}")
            s4.metric("LFCC std",  f"{lfcc.std():.3f}")
            s5.metric("Range",     f"{lfcc.max()-lfcc.min():.3f}")

        with tab3:
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("#### 🔬 What is LFCC?")
                st.markdown("""
**Linear Frequency Cepstral Coefficients** — a compact audio fingerprint.

**5-step pipeline (Cell 4):**
1. **STFT** → 94 windows × 257 freq bins
2. **Linear filterbank** → 70 equal filters
3. **Log** → compress dynamic range
4. **DCT** → keep top 40 coefficients
5. **Pad/truncate** → fixed (40 × 94)

**Why linear not mel?**
Linear scale preserves high-frequency detail where AI vocoders leave artefacts — MFCC's log scale would compress them away.
                """)

            with col_b:
                st.markdown("#### 🧠 LCNN architecture (Cell 10)")
                st.code("""
Input (40, 94, 1)
  ↓
Conv(64) + MFM → 32ch
BatchNorm + Pool + Drop
  → (20, 47, 32)
  ↓
Conv(128) + MFM → 64ch
BatchNorm + Pool + Drop
  → (10, 23, 64)
  ↓
Conv(256) + MFM → 128ch
BatchNorm + Pool + Drop
  → (5, 11, 128)
  ↓
Flatten (7040)
Dense(256) + MFM → 128
Dropout(0.3)
  ↓
Dense(1, sigmoid)
0.0=REAL · 1.0=FAKE
""", language="text")

            with col_c:
                st.markdown("#### 🏆 Why LCNN won")
                st.markdown("""
**MFM vs ReLU:**
- ReLU: `max(0, x)` — zeros negatives
- MFM: `max(x[:N], x[N:])` — competition

MFM forces **two feature maps to compete** — only the stronger signal survives. Better noise suppression, better at capturing subtle AI artefacts.

**Final results (Cell 13):**
| Metric | Result | Target |
|---|---|---|
| Accuracy | **88.51%** | ≥ 80% ✓ |
| EER | **11.58%** | ≤ 12% ✓ |
| F1 Score | **88.52%** | ≥ 80% ✓ |
| Real Acc | **88.42%** | ≥ 75% ✓ |
| Fake Acc | **88.60%** | ≥ 75% ✓ |
                """)

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        st.info("Make sure the file is a valid .wav containing speech.")

    finally:
        os.unlink(tmp_path)

else:
    st.info("👆 Upload a .wav file above to get started", icon="ℹ️")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Works best with:**")
        st.markdown("""
- `.wav` format · 2–5 seconds
- Clear speech · single speaker
- 16 kHz sample rate
        """)
    with c2:
        st.markdown("**⚠️ Less accurate with:**")
        st.markdown("""
- Music or silence
- Heavy background noise
- Very short clips (< 1s)
        """)

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.info("🎯 **Verdict** — REAL or FAKE with confidence %")
    col2.info("🔬 **LFCC heatmap** — 40×94 frequency fingerprint")
    col3.info("📊 **Statistics** — coefficient distribution + frame energy")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
f1, f2, f3 = st.columns(3)
f1.caption("🏛️ IIT Roorkee · MARS Club Open Project 2026")
f2.caption("👤 Sameer Modi (23410030) · Geological Technology")
f3.caption("🏆 LCNN + MFM + LFCC · 88.51% Accuracy · EER 11.58%")