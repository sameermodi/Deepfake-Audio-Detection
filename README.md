# 🎙️ Deepfake Audio Detection
### *Can you tell if a voice is real? Our AI can.*

> **MARS Club Open Project 2026 — IIT Roorkee**
> Built by **Sameer Modi** (23410030) · Geological Technology

---

## 🚨 The Problem

Imagine getting a voice message from your dad asking you to transfer money urgently.
But what if it wasn't really him?

AI can now **clone any voice** in seconds. Deepfake audio is being used for:
- 💸 Financial fraud and scams
- 🎭 Impersonation attacks
- 📰 Misinformation and fake news
- 🔐 Breaking voice authentication systems

**This project builds a detector that can tell the difference.**

---

## ✅ Results — ALL 5 Criteria Passed

| Metric | Target | Our Result | Status |
|:---|:---:|:---:|:---:|
| 🎯 Overall Accuracy | ≥ 80% | **88.51%** | ✅ PASS |
| 📉 Equal Error Rate (EER) | ≤ 12% | **11.58%** | ✅ PASS |
| 📊 F1 Score | ≥ 80% | **88.52%** | ✅ PASS |
| 🟢 Real Voice Accuracy | ≥ 75% | **88.42%** | ✅ PASS |
| 🔴 Fake Voice Accuracy | ≥ 75% | **88.60%** | ✅ PASS |

> 🏆 **LCNN with MFM activation achieved all 5 MARS Club verification criteria**

---

## 🧠 How It Works

```
Your audio file
      │
      ▼
┌─────────────────────────────┐
│   LFCC Feature Extraction   │  ← converts audio to a 40×94
│   (Linear Freq. Cepstral    │    frequency fingerprint
│    Coefficients)            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   LCNN with MFM Activation  │  ← Light CNN scans the
│   (Light CNN)               │    fingerprint for AI artefacts
│                             │
│  Conv(64)+MFM  → (20×47)   │
│  Conv(128)+MFM → (10×23)   │
│  Conv(256)+MFM → (5×11)    │
│  Dense(256)+MFM → 128      │
│  Dense(1) → sigmoid         │
└─────────────┬───────────────┘
              │
              ▼
      REAL ✅ or FAKE 🚨
     + confidence score
```

### 🔬 What is LFCC?

Unlike regular MFCC (which uses logarithmic mel scale for human hearing),
**LFCC uses linear frequency spacing** — which preserves high-frequency detail
where AI vocoders leave telltale artefacts.

Each audio clip becomes a **40×94 grid** — like a fingerprint of the sound.
Real voices and AI voices look visually different in this grid.
The CNN's job is to learn that difference.

### ⚡ What is MFM (Max Feature Map)?

| | Regular CNN (ReLU) | Our LCNN (MFM) |
|---|---|---|
| How | `max(0, x)` | `max(x[:N], x[N:])` |
| Competition | ❌ None | ✅ Two maps compete |
| Noise suppression | Moderate | Strong |
| Best for | General vision | Anti-spoofing |

MFM forces **two feature maps to compete** — only the strongest signal survives.
This makes it much better at catching subtle AI artefacts.

---

## 📊 CNN vs LCNN — Head to Head

We trained both architectures on the same data and let the metrics decide.

| Metric | CNN Baseline | LCNN + MFM ✓ | Winner |
|:---|:---:|:---:|:---:|
| Accuracy | 86.49% | **88.51%** | 🏆 LCNN |
| EER (lower = better) | 13.24% | **11.58%** | 🏆 LCNN |
| F1 Score | 86.27% | **88.52%** | 🏆 LCNN |
| Real Accuracy | 88.05% | **88.42%** | 🏆 LCNN |
| Fake Accuracy | 84.93% | **88.60%** | 🏆 LCNN |

> **LCNN won all 5 metrics → selected for deployment**

---

## 🔊 Data Augmentation

To make the model robust to real-world conditions,
we created **3 variations of every training file**:

| Technique | What it simulates | Parameter |
|---|---|---|
| 🎤 Gaussian noise | Microphone imperfections | factor = 0.005 |
| ⏩ Time stretch | Different speaking speeds | rate = 1.1 (10% faster) |
| 🔊 Gain variation | Different recording volumes | ±20% |

**Result:** 13,956 files → **~55,824 training samples**

---

## 📦 Dataset

**[The Fake-or-Real (FoR) Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)**
by Mohammed Abdel Dayem — Kaggle

> *"A collection of more than 195,000 utterances from real humans and computer generated speech."*

We used the **for-2sec version** — fixed 2-second clips, perfectly balanced, already normalised.

| Split | Real | Fake | Total |
|---|---|---|---|
| Training | 6,978 | 6,978 | 13,956 |
| Validation | 1,413 | 1,413 | 2,826 |
| Testing | 544 | 544 | 1,088 |
| **Total** | **8,935** | **8,935** | **17,870** |

**Why for-2sec version?**
- ✅ Fixed 2-second clips — no padding variation
- ✅ Perfectly balanced — equal real and fake
- ✅ Already normalised and silence-trimmed
- ✅ Manageable size — fast to train on Kaggle GPU

**Fake sources:** Deep Voice 3, Google WaveNet TTS
**Real sources:** Arctic, LJSpeech, VoxForge datasets

**Dataset versions available:**

| Version | Description | Used? |
|---|---|---|
| for-original | Raw files as collected | — |
| for-norm | Normalised for gender and class | — |
| **for-2sec** | Fixed 2-second clips | ✅ This project |
| for-rerec | Re-recorded version | — |

🔗 **[Download from Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)**

---

## 🗺️ Training Pipeline (8 Phases)

| Phase | Description |
|:---:|---|
| 1️⃣ | Environment setup and data exploration |
| 2️⃣ | LFCC feature extraction (40×94 per file) |
| 3️⃣ | Data augmentation (noise + stretch + gain) |
| 4️⃣ | Normalization and channel dimension prep |
| 5️⃣ | CNN baseline — build and train |
| 6️⃣ | LCNN with MFM activation — build and train |
| 7️⃣ | Model comparison and evaluation |
| 8️⃣ | Save best model for Streamlit deployment |

---

## 🎯 Confusion Matrix (Test Set — 1,088 files)

```
                  Predicted
               Real    Fake
Actual  Real  [ 481  |  63 ]   → 88.42% correct ✅
        Fake  [  62  | 482 ]   → 88.60% correct ✅
```

- 481 real voices correctly identified ✅
- 482 deepfakes correctly caught 🚨
- Only 125 total errors out of 1,088 files

---

## 📁 Project Structure

```
Deepfake-Audio-Detection/
│
├── 🐍 app.py                  # Streamlit web app
├── 🧠 best_model.h5           # Trained LCNN model (~23 MB)
├── 📦 norm_params.pkl         # Normalization parameters (mean + std)
├── 🎯 threshold.pkl           # Optimal decision threshold (0.050)
├── 📓 notebook.ipynb          # Full training pipeline (8 phases)
├── 📊 performance_report.png  # CNN vs LCNN visual comparison
└── 📋 requirements.txt        # Python dependencies
```

---

## 🚀 Run Locally

**Step 1 — Clone the repo**
```bash
git clone https://github.com/sameermodi/Deepfake-Audio-Detection.git
cd Deepfake-Audio-Detection
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Launch the app**
```bash
streamlit run app.py
```

**Step 4 — Upload any `.wav` file and get your verdict!**

---

## 🌐 Try It Live

👉 **[Launch the Web App](https://your-app.streamlit.app)**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| 🐍 Python 3.12 | Core language |
| 🧠 TensorFlow / Keras | LCNN model building and training |
| 🔢 NumPy + SciPy | LFCC feature extraction |
| 🔊 soundfile | Audio file loading |
| 📊 scikit-learn | Metrics — accuracy, F1, EER, confusion matrix |
| 📈 Matplotlib | Visualisations and plots |
| 🌐 Streamlit | Web app deployment |
| ☁️ Kaggle GPU T4 ×2 | Model training environment |

---

## 👨‍💻 Author

| | |
|---|---|
| **Name** | Sameer Modi |
| **Roll No** | 23410030 |
| **Branch** | Geological Technology |
| **Institute** | IIT Roorkee |
| **Club** | MARS Club — Open Project 2026 |
