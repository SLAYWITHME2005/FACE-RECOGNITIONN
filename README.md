# DOSTH AI — Facial Expression Recognition for Elderly Care

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/ONNX%20Runtime-CPU-orange" />
  <img src="https://img.shields.io/badge/OpenCV-YuNet%20Face%20Detector-green" />
  <img src="https://img.shields.io/badge/FastAPI-Web%20Dashboard-teal" />
  <img src="https://img.shields.io/badge/Tests-21%20passing-brightgreen" />
</p>

**DOSTH AI** is an elderly-accessible, real-time Facial Expression Recognition (FER) system built for caregiving environments. It uses a live webcam feed, deep-learning face detection, and a FER+ ONNX model to predict facial expressions — with a calm, readable web dashboard designed specifically for elderly users.

---

## 🧠 What It Does

| Feature | Details |
|---|---|
| **Live Camera Feed** | Streams real-time webcam video with face bounding box overlay |
| **Face Detection** | YuNet deep-learning detector (ONNX, CPU-only) via OpenCV |
| **Expression Classification** | FER+ ONNX model — 7 emotions: Neutral, Happy, Sad, Angry, Fear, Surprise, Disgust |
| **Temporal Smoothing** | Exponential Moving Average (EMA) + hysteresis to prevent flickering labels |
| **Confidence Tiers** | Honest three-tier output: High / Moderate / Unclear |
| **Elderly-Accessible UI** | Large text, dark calm theme, simple controls, keyboard-navigable |
| **Self-Report Feedback** | "How are you feeling?" voluntary check-in buttons |
| **Environmental Guidance** | Detects low light, blur, face tilt, and guides user to adjust |
| **Caregiver Dashboard** | Expression history, confidence over time, wellbeing check-in log |

---

## 📁 Project Structure

```
dosth-ai/
│
├── modules/
│   └── fer/
│       ├── emotion_model.py      # FER+ ONNX inference + geometric landmark hints
│       ├── face_detector.py      # YuNet face detector with EMA bounding-box smoothing
│       ├── image_enhancer.py     # CLAHE + bilateral filter preprocessing
│       ├── pipeline.py           # Full camera → face → classify → smooth pipeline
│       └── elderly_adapter.py   # Optional fine-tuned elderly expression adapter
│
├── scripts/
│   ├── run_fer_web_ui.py         # FastAPI web dashboard (main entry point)
│   ├── run_fer_webcam.py         # Standalone OpenCV webcam viewer (no browser needed)
│   ├── collect_training_data.py  # Collect labelled face images from webcam
│   ├── finetune_elderly.py       # Fine-tune the adapter layer on collected data
│   ├── setup_elderly_model.py    # One-time setup: download ONNX models
│   └── diagnose_live_happy.py    # Diagnostic tool to debug prediction bias
│
├── tests/
│   ├── test_elderly_fer_system.py  # Core system: confidence tiers, smoothing, stale state
│   ├── test_fer.py                 # Pipeline: blank frame, temporal smoothing, shapes
│   ├── test_5_emotions_accuracy.py # Geometric accuracy: 5 core emotions
│   ├── test_angry_fear_sad.py      # Landmark activation: Angry, Fear, Sad, Neutral, Happy
│   └── test_camera_diagnostic.py  # Camera + pipeline integration check
│
├── data/
│   └── collected/elderly_faces/    # Sample collected face images per emotion class
│
├── models/
│   └── fer/                        # Auto-downloaded ONNX models (not in repo)
│       ├── emotion_model.onnx      # FER+ expression classifier
│       └── face_detection_yunet.onnx # YuNet face detector
│
├── run_dosth_ai.bat              # One-click launcher (Windows)
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 How to Run

### Requirements
- **Windows 10/11** (tested), Python **3.10+**
- Webcam connected
- Internet access on first run (to auto-download ONNX models ~50 MB)

---

### Option 1 — One-Click (Windows)

Double-click **`run_dosth_ai.bat`** in File Explorer.

Then open your browser at:
```
http://localhost:8000
```

---

### Option 2 — Command Line

**Step 1: Create and activate virtual environment**
```powershell
cd dosth-ai
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Step 2: Install dependencies**
```powershell
pip install -r requirements.txt
pip install fastapi uvicorn
```

**Step 3: Run the web dashboard**
```powershell
python scripts/run_fer_web_ui.py
```

Open **http://localhost:8000** in your browser.

---

### Option 3 — Standalone Webcam Viewer (no browser)

```powershell
python scripts/run_fer_webcam.py
```

Press `Q` to quit.

---

## 🎯 Emotions Detected

| Emotion | Triggered By |
|---|---|
| 😐 **Neutral** | Resting face, no significant mouth movement |
| 😊 **Happy** | Mouth widens ≥3.8% of inter-ocular distance |
| 😢 **Sad** | Mouth corners droop downward, mouth stays closed |
| 😠 **Angry** | Lips press/purse together (mouth narrows) |
| 😨 **Fear** | Mouth opens wide AND jaw drops (open + horizontal stretch) |
| 😲 **Surprise** | Jaw drops, mouth fully open |
| 🤢 **Disgust** | Model-detected + FER+ class combination |

---

## 🧪 Running Tests

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

Expected output:
```
21 passed in ~20s ✅
```

| Test File | What It Tests |
|---|---|
| `test_elderly_fer_system.py` | Confidence tiers, EMA smoothing, stale-state clearing |
| `test_fer.py` | Pipeline basics, shapes, blank-frame safety |
| `test_5_emotions_accuracy.py` | All 5 core emotions via geometric landmark simulation |
| `test_angry_fear_sad.py` | Angry/Fear/Sad/Neutral/Happy activation — geometric precision |
| `test_camera_diagnostic.py` | End-to-end camera + pipeline integration |

---

## ⚙️ How It Works (Pipeline)

```
Webcam Frame
    │
    ▼
YuNet Face Detector  ──► No face → "Please look at camera" guidance
    │
    ▼
Image Enhancement (CLAHE + bilateral filter)
    │
    ▼
Face Crop (tight 6% margin → 64×64 grayscale)
    │
    ▼
FER+ ONNX Model Inference  →  8 raw logits
    │
    ▼
Geometric Landmark Hints (mouth width/height vs resting baseline)
  • Resting: Neutral +2.0 boost
  • Smile  : Happy boost (dW > 0.038)
  • Angry  : Pursed lips boost (dW < -0.020)
  • Fear   : Jaw drop + wide (dH > 0.030 & dW > 0.012)
  • Sad    : Droop, closed mouth (dH > 0.028 & dW ≤ 0.012)
    │
    ▼
Temperature Softmax → 7-class probabilities
    │
    ▼
EMA Temporal Smoothing (α=0.35) + Hysteresis (margin=0.08)
    │
    ▼
Confidence Tier: High / Moderate / Unclear
    │
    ▼
Web Dashboard Update
```

---

## 🛡️ Key Design Decisions

### Why FER+ ONNX?
- Runs entirely on **CPU** — no GPU required
- No scipy, sklearn, or heavy ML libraries (Windows WDAC compatible)
- Fast enough for real-time webcam inference (~15–30 FPS)

### Why geometric landmark hints?
The raw FER+ model was trained on internet images and has a bias toward Happy and Neutral on resting elderly faces. Mouth-geometry hints correct this without retraining:
- `neutral_suppression = 0.0` — never artificially demote Neutral
- Baseline subtraction only targets Sad/Angry resting logits, not Neutral
- Face crop margin `= 6%` (not 22%) — keeps the face tight in the 64×64 input

### Why EMA + Hysteresis?
Prevents label flickering between frames without locking to one emotion permanently. A new emotion needs to dominate by ≥8% (`hysteresis_margin`) or persist for 3 consecutive frames before the label switches.

---

## 🧓 Accessibility Features

- **Large, readable text** (≥18px minimum, scalable with browser zoom)
- **Dark calm theme** — no flashing or rapid colour changes
- **Plain-language guidance** — "Room lighting is dim. Turning on a nearby light may help."
- **Keyboard navigable** — all buttons accessible without mouse
- **Screen-reader friendly** — meaningful status changes announced, not every frame
- **Voluntary self-report** — "How are you feeling?" with simple emoji options
- **No alarming red banners** — replaced with calm supportive wellbeing check-ins

---

## ⚠️ Limitations & Honest Notes

- Predictions are based on **visible facial appearance**, not actual feelings or mental state
- The model is not calibrated — confidence scores are not true probabilities
- Accuracy varies with lighting, camera angle, face distance, and occlusion
- This is **not a medical device** and should not be used for clinical diagnosis
- Elderly faces (wrinkles, different skin tone) may affect raw model accuracy

---

## 🔧 Collecting Your Own Training Data

To improve accuracy for a specific person:

```powershell
python scripts/collect_training_data.py
```

Then fine-tune the adapter:
```powershell
python scripts/finetune_elderly.py
```

This trains a lightweight adapter layer on top of the FER+ model using only the collected images.

---

## 📦 Dependencies

```
numpy>=1.26.0
opencv-python>=4.8.0
pillow>=10.0.0
onnxruntime>=1.18.0
fastapi
uvicorn
```

> **Note:** `scipy` is listed in requirements.txt but is not used in the core pipeline (WDAC policy compatible). Only numpy, opencv, onnxruntime, and fastapi are required for normal operation.

---

## 👩‍💻 Authors

**DOSTH AI** — Designed as part of a Multimodal Emotion-Aware AI Assistant for Elderly Care.

- FER module: YuNet + FER+ ONNX with geometric landmark correction
- UI: FastAPI + vanilla HTML/JS, elderly-accessible design
- Tests: 21 automated unit + integration tests

---

## 📄 License

This project is for educational and research purposes. The FER+ ONNX model is sourced from [Microsoft's FERPlus dataset](https://github.com/microsoft/FERPlus) and [ONNX Model Zoo](https://github.com/onnx/models).
