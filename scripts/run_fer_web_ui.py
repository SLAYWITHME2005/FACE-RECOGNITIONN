import cv2
import time
import numpy as np
import os
import sys
import webbrowser
import threading
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

# Ensure UTF-8 output on Windows to prevent encoding crashes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.fer.pipeline import FacialEmotionPipeline

app = FastAPI(title="DOSTH AI - Elderly-Accessible Facial Expression Assistant")

pipeline = FacialEmotionPipeline()
camera_lock = threading.Lock()
cap = None
camera_running = True
enable_enhancement = True
user_self_reports = []

latest_result = {
    "face_detected": False,
    "primary_emotion": "Neutral",
    "confidence": 0.0,
    "confidence_level": "Unclear",
    "is_confident": False,
    "display_label": "No face detected",
    "status_message": "Camera starting up. Please sit facing the camera.",
    "guidance_message": "Please look directly at the camera.",
    "icon": "👤",
    "probabilities": {"Happy": 0, "Neutral": 100, "Surprise": 0, "Sad": 0, "Angry": 0, "Fear": 0, "Disgust": 0},
    "valence": 0.0,
    "arousal": 0.0,
    "support_checkin": False,
    "support_message": "",
    "quality_metrics": {"brightness": 0, "sharpness": 0},
    "fps": 0.0,
    "camera_running": True,
    "enhancement_active": True,
    "baseline_calibrated": False,
    "calibration_progress": 0.0,
    "geo_debug": {"dW": None, "dH": None, "is_resting": False, "geo_active": False}
}

EMOTION_COLORS = {
    'Happy':    (0, 220, 100),
    'Neutral':  (230, 210, 160),
    'Surprise': (0, 215, 255),
    'Sad':      (255, 140, 0),
    'Fear':     (200, 50, 180),
    'Angry':    (40, 40, 240),
    'Disgust':  (50, 140, 90)
}

def get_camera():
    global cap, camera_running
    if not camera_running:
        return None
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def release_camera():
    global cap, camera_running
    camera_running = False
    with camera_lock:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None

def create_paused_frame(width=1280, height=720):
    """Generates a calm, dignified placeholder frame when camera is paused."""
    placeholder = np.full((height, width, 3), 18, dtype=np.uint8)
    text = "Camera is paused"
    subtext = "Click 'Start Camera' below to resume"
    
    cv2.putText(placeholder, text, (width // 2 - 220, height // 2 - 20),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (200, 215, 230), 2, cv2.LINE_AA)
    cv2.putText(placeholder, subtext, (width // 2 - 280, height // 2 + 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (120, 145, 170), 1, cv2.LINE_AA)
    return placeholder

def generate_frames():
    global latest_result, enable_enhancement, camera_running
    prev_time = time.time()
    fps = 0.0

    while True:
        if not camera_running:
            # Generate calm placeholder frame
            frame = create_paused_frame()
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.2)
            continue

        camera = get_camera()
        if camera is None or not camera.isOpened():
            frame = create_paused_frame()
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.3)
            continue

        with camera_lock:
            ret, frame = camera.read()
            
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)  # Mirror for natural self-view
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
        prev_time = curr_time

        result = pipeline.process_frame(frame, enable_enhancement=enable_enhancement)
        
        detected = result.get("face_detected", False)
        primary_emo = result.get("primary_emotion", "Neutral")
        conf = float(result.get("confidence", 0.0))
        conf_level = result.get("confidence_level", "Moderate")
        is_conf = result.get("is_confident", True)
        display_lbl = result.get("display_label", primary_emo)
        status_msg = result.get("status_message", "")
        guidance_msg = result.get("guidance_message", "")
        icon = result.get("icon", "😐")
        probs = result.get("probabilities", {})
        valence = float(result.get("valence", 0.0))
        arousal = float(result.get("arousal", 0.0))
        support_checkin = bool(result.get("support_checkin", False))
        support_msg = result.get("support_message", "")
        q_metrics = result.get("quality_metrics") or {"brightness": 0, "sharpness": 0}

        latest_result = {
            "face_detected": detected,
            "primary_emotion": primary_emo,
            "confidence": round(conf * 100, 1),
            "confidence_level": conf_level,
            "is_confident": is_conf,
            "display_label": display_lbl,
            "status_message": status_msg,
            "guidance_message": guidance_msg,
            "icon": icon,
            "probabilities": {k: round(v * 100, 1) for k, v in probs.items()},
            "valence": valence,
            "arousal": arousal,
            "support_checkin": support_checkin,
            "support_message": support_msg,
            "quality_metrics": q_metrics,
            "fps": round(fps, 1),
            "camera_running": camera_running,
            "enhancement_active": enable_enhancement,
            "baseline_calibrated": bool(result.get("baseline_calibrated", False)),
            "calibration_progress": float(result.get("calibration_progress", 0.0)),
            "geo_debug": result.get("geo_debug", {})
        }

        # ── Draw Face Bounding Box & Honest Emotion Badge ────────────────────
        if detected and result.get("bounding_box"):
            x, y, fw, fh = result["bounding_box"]
            color = EMOTION_COLORS.get(primary_emo, (220, 220, 220))

            # Outer bounding box
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
            
            # Corner accents
            c_len = int(min(fw, fh) * 0.20)
            cv2.line(frame, (x, y), (x + c_len, y), color, 4)
            cv2.line(frame, (x, y), (x, y + c_len), color, 4)
            cv2.line(frame, (x + fw, y), (x + fw - c_len, y), color, 4)
            cv2.line(frame, (x + fw, y), (x + fw, y + c_len), color, 4)
            cv2.line(frame, (x, y + fh), (x + c_len, y + fh), color, 4)
            cv2.line(frame, (x, y + fh), (x, y + fh - c_len), color, 4)
            cv2.line(frame, (x + fw, y + fh), (x + fw - c_len, y + fh), color, 4)
            cv2.line(frame, (x + fw, y + fh), (x + fw, y + fh - c_len), color, 4)

            # Honest overlay badge
            if not is_conf:
                badge = " EXPRESSION UNCLEAR "
                bg_color = (80, 80, 80)
            elif conf_level == "Moderate":
                badge = f" POSSIBLE: {primary_emo.upper()} ({int(conf * 100)}%) "
                bg_color = color
            else:
                badge = f" {primary_emo.upper()}: {int(conf * 100)}% "
                bg_color = color

            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_DUPLEX, 0.70, 2)
            by = max(0, y - th - 12)
            cv2.rectangle(frame, (x, by), (x + tw + 10, y), bg_color, -1)
            cv2.putText(frame, badge, (x + 5, y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/telemetry")
def get_telemetry():
    return latest_result

@app.post("/api/camera/start")
def start_camera():
    global camera_running
    camera_running = True
    get_camera()
    latest_result["camera_running"] = True
    return {"status": "started", "message": "Camera is active"}

@app.post("/api/camera/stop")
def stop_camera():
    release_camera()
    latest_result["camera_running"] = False
    latest_result["face_detected"] = False
    latest_result["display_label"] = "Camera paused"
    latest_result["status_message"] = "Camera is currently paused."
    return {"status": "stopped", "message": "Camera released"}

@app.post("/api/toggle_enhancer")
def toggle_enhancer():
    global enable_enhancement
    enable_enhancement = not enable_enhancement
    return {"enhancement_active": enable_enhancement}

@app.post("/api/reset_user")
def reset_user():
    pipeline.reset_for_new_user()
    latest_result["baseline_calibrated"] = False
    latest_result["calibration_progress"] = 0.0
    return {"status": "baseline_reset", "message": "Calibration restarted for new user"}

@app.post("/api/user_feedback")
async def user_feedback(request: Request):
    try:
        data = await request.json()
        feeling = data.get("feeling", "Unspecified")
        entry = {
            "feeling": feeling,
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S")
        }
        user_self_reports.append(entry)
        return {"status": "recorded", "message": f"Thank you! We noted that you are feeling: {feeling}"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ── Accessible Elderly-Friendly Frontend ───────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DOSTH AI - Elderly Facial Expression Assistant</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0b0f19;
                --card-bg: #131b2e;
                --border: #1e293b;
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --primary-blue: #38bdf8;
                --accent-green: #22c55e;
                --accent-amber: #f59e0b;
                --accent-purple: #a855f7;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
            body { background: var(--bg); color: var(--text-primary); padding: 24px; min-height: 100vh; line-height: 1.5; }

            /* Focus indicator for keyboard accessibility */
            *:focus-visible {
                outline: 3px solid var(--primary-blue);
                outline-offset: 3px;
                border-radius: 6px;
            }

            .container { max-width: 1200px; margin: 0 auto; }

            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 16px;
                border-bottom: 1px solid var(--border);
                flex-wrap: wrap;
                gap: 16px;
            }

            .logo-group h1 {
                font-size: 28px;
                font-weight: 800;
                color: var(--primary-blue);
                letter-spacing: -0.5px;
            }

            .logo-group p {
                color: var(--text-secondary);
                font-size: 15px;
            }

            .status-indicators {
                display: flex;
                gap: 12px;
                align-items: center;
                flex-wrap: wrap;
            }

            .pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 16px;
                border-radius: 9999px;
                font-size: 14px;
                font-weight: 700;
            }
            .pill-active { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
            .pill-paused { background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); }
            .pill-guidance { background: rgba(56, 189, 248, 0.12); color: #93c5fd; border: 1px solid rgba(56, 189, 248, 0.25); }

            /* Calm Wellbeing Support Banner */
            .support-banner {
                display: none;
                background: linear-gradient(135deg, #1e1b4b, #312e81);
                border: 1px solid #6366f1;
                color: #e0e7ff;
                padding: 16px 20px;
                border-radius: 14px;
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 20px;
                box-shadow: 0 4px 20px rgba(99, 102, 241, 0.2);
            }

            .calib-banner {
                background: #1e293b;
                border: 1px solid #38bdf8;
                color: #93c5fd;
                padding: 14px 18px;
                border-radius: 12px;
                font-size: 15px;
                font-weight: 600;
                margin-bottom: 20px;
            }
            .calib-bar-track { height: 8px; background: #0f172a; border-radius: 6px; overflow: hidden; margin-top: 8px; }
            .calib-bar-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #38bdf8, #818cf8); transition: width 0.3s ease; }

            .main-grid {
                display: grid;
                grid-template-columns: 1.5fr 1fr;
                gap: 24px;
            }
            @media (max-width: 950px) { .main-grid { grid-template-columns: 1fr; } }

            .card {
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 22px;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
            }

            .video-box {
                position: relative;
                border-radius: 14px;
                overflow: hidden;
                background: #020617;
                aspect-ratio: 16/9;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 1px solid #1e293b;
            }
            .video-box img { width: 100%; height: 100%; object-fit: cover; }

            .controls-row {
                display: flex;
                gap: 12px;
                margin-top: 18px;
                flex-wrap: wrap;
            }

            .btn {
                background: #2563eb;
                color: #ffffff;
                border: none;
                padding: 12px 22px;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 700;
                cursor: pointer;
                transition: background 0.15s ease, transform 0.1s ease;
                min-height: 48px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }
            .btn:hover { background: #1d4ed8; }
            .btn:active { transform: scale(0.98); }
            .btn-outline { background: #1e293b; border: 1px solid #334155; color: #f1f5f9; }
            .btn-outline:hover { background: #334155; }
            .btn-danger { background: #dc2626; }
            .btn-danger:hover { background: #b91c1c; }

            /* Primary Expression Card */
            .expression-card {
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }

            .expression-hero {
                background: #0f172a;
                border-radius: 14px;
                padding: 22px;
                text-align: center;
                border: 1px solid #1e293b;
                margin-bottom: 20px;
            }

            .hero-icon { font-size: 56px; line-height: 1; margin-bottom: 10px; }
            .hero-title { font-size: 13px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
            .hero-emotion { font-size: 32px; font-weight: 800; color: #ffffff; margin: 4px 0 8px 0; }
            
            .confidence-pill {
                display: inline-block;
                padding: 6px 14px;
                border-radius: 9999px;
                font-size: 14px;
                font-weight: 700;
                background: rgba(56, 189, 248, 0.15);
                color: #38bdf8;
                border: 1px solid rgba(56, 189, 248, 0.3);
            }

            .expression-disclaimer {
                font-size: 13px;
                color: var(--text-secondary);
                margin-top: 10px;
                line-height: 1.4;
            }

            /* Self-Report Check-In Section */
            .checkin-section {
                background: #0f172a;
                border-radius: 14px;
                padding: 18px;
                border: 1px solid #1e293b;
                margin-top: 14px;
            }
            .checkin-title {
                font-size: 15px;
                font-weight: 700;
                color: #f1f5f9;
                margin-bottom: 12px;
            }
            .feeling-buttons {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
                gap: 8px;
            }
            .btn-feeling {
                background: #131b2e;
                border: 1px solid #334155;
                color: #f8fafc;
                padding: 10px 8px;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 700;
                cursor: pointer;
                text-align: center;
                transition: 0.15s;
            }
            .btn-feeling:hover {
                background: #1e293b;
                border-color: #38bdf8;
                color: #38bdf8;
            }
            .feeling-feedback {
                margin-top: 10px;
                font-size: 13px;
                color: #4ade80;
                font-weight: 600;
                display: none;
            }

            /* Collapsible Technical Details */
            details {
                margin-top: 24px;
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 16px 20px;
            }
            summary {
                font-size: 15px;
                font-weight: 700;
                color: #94a3b8;
                cursor: pointer;
                user-select: none;
            }
            summary:hover { color: #f1f5f9; }

            .tech-content {
                margin-top: 16px;
                padding-top: 16px;
                border-top: 1px solid #1e293b;
            }

            .prob-row { margin-bottom: 10px; }
            .prob-labels { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; color: #cbd5e1; }
            .progress-track { height: 8px; background: #0f172a; border-radius: 6px; overflow: hidden; }
            .progress-fill { height: 100%; width: 0%; border-radius: 6px; transition: width 0.25s ease; }

            .stat-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
                gap: 10px;
                margin-top: 16px;
            }
            .stat-box {
                background: #0f172a;
                padding: 10px 14px;
                border-radius: 8px;
                border: 1px solid #1e293b;
            }
            .stat-label { font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; }
            .stat-val { font-size: 16px; font-weight: 700; color: #f8fafc; margin-top: 2px; }

            .privacy-note {
                margin-top: 24px;
                text-align: center;
                font-size: 13px;
                color: #64748b;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <header class="header">
                <div class="logo-group">
                    <h1>DOSTH AI</h1>
                    <p>Elderly-Accessible Facial Expression Assistant</p>
                </div>
                <div class="status-indicators">
                    <span id="cameraPill" class="pill pill-active">● Camera Active</span>
                    <span id="guidancePill" class="pill pill-guidance">💡 Good Lighting</span>
                </div>
            </header>

            <!-- Calm Support Check-in (Replaces alarming red alert) -->
            <div id="supportBanner" class="support-banner" role="region" aria-live="polite">
                <div style="display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap;">
                    <span id="supportText">Checking in: We notice you might be having a difficult moment. Would you like to take a break or talk to someone?</span>
                    <button class="btn btn-outline" style="min-height: 38px; padding: 6px 14px; font-size: 13px;" onclick="dismissSupport()">I am okay, thank you</button>
                </div>
            </div>

            <!-- Calibration Banner -->
            <div id="calibBanner" class="calib-banner" role="status">
                ⏳ <b>Setting up your personal baseline</b> — Please look comfortably at the camera for a moment...
                <div class="calib-bar-track"><div class="calib-bar-fill" id="calibBarFill"></div></div>
            </div>

            <!-- Main Layout Grid -->
            <main class="main-grid">
                <!-- Left: Live Camera Card -->
                <section class="card" aria-label="Camera Feed and Controls">
                    <h2 style="font-size: 17px; margin-bottom: 12px; color: #f1f5f9;">Live Video View</h2>
                    <div class="video-box">
                        <img id="videoStream" src="/video_feed" alt="Live Camera Video Stream">
                    </div>

                    <div class="controls-row">
                        <button class="btn" id="btnCameraToggle" onclick="toggleCamera()" aria-label="Pause or start the camera feed">
                            Pause Camera
                        </button>
                        <button class="btn btn-outline" id="btnEnhancer" onclick="toggleEnhancer()" aria-label="Toggle low light image enhancement">
                            Light Enhancer: ON
                        </button>
                        <button class="btn btn-outline" onclick="resetUser()" aria-label="Reset baseline calibration for a new person">
                            New Person Setup
                        </button>
                    </div>

                    <p id="envGuidanceText" style="margin-top: 14px; font-size: 14px; color: #94a3b8;">
                        Tip: Sit comfortably in a well-lit room facing the camera.
                    </p>
                </section>

                <!-- Right: Current Expression Card -->
                <section class="card expression-card" aria-label="Current Facial Expression Result">
                    <div>
                        <div class="expression-hero" role="region" aria-live="polite">
                            <div class="hero-icon" id="heroIcon">👤</div>
                            <div class="hero-title">Visible Facial Expression</div>
                            <div class="hero-emotion" id="heroEmotion">Checking...</div>
                            <div>
                                <span class="confidence-pill" id="heroConfidence">Confidence: Waiting</span>
                            </div>
                            <p id="heroStatusMsg" style="margin-top: 10px; font-size: 14px; color: #cbd5e1;">
                                Looking for your face...
                            </p>
                        </div>

                        <p class="expression-disclaimer">
                            ℹ️ Facial expression estimates are automated visual predictions and may not always reflect how you truly feel.
                        </p>
                    </div>

                    <!-- User Self-Report ("How are you feeling?") -->
                    <div class="checkin-section" aria-label="Self-Report Emotional State">
                        <div class="checkin-title">How are you feeling right now?</div>
                        <div class="feeling-buttons">
                            <button class="btn-feeling" onclick="sendFeeling('Happy / Good')">😊 Good</button>
                            <button class="btn-feeling" onclick="sendFeeling('Neutral / Okay')">😐 Okay</button>
                            <button class="btn-feeling" onclick="sendFeeling('A bit down')">😔 Down</button>
                            <button class="btn-feeling" onclick="sendFeeling('Anxious / Worried')">😟 Anxious</button>
                            <button class="btn-feeling" onclick="sendFeeling('Frustrated')">😠 Upset</button>
                        </div>
                        <div id="feelingFeedback" class="feeling-feedback"></div>
                    </div>
                </section>
            </main>

            <!-- Collapsible Technical Details (For Caregivers & Engineers) -->
            <details>
                <summary>View Technical Details & Probability Distribution</summary>
                <div class="tech-content">
                    <h3 style="font-size: 14px; color: #94a3b8; margin-bottom: 12px; text-transform: uppercase;">Real-Time Expression Probabilities</h3>
                    <div id="probContainer"></div>

                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-label">Camera FPS</div>
                            <div class="stat-val" id="statFps">0.0</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Room Lighting</div>
                            <div class="stat-val" id="statLight">0</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Focus Sharpness</div>
                            <div class="stat-val" id="statSharp">0</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Micro-Movement (dW)</div>
                            <div class="stat-val" id="statDW">0.0000</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Micro-Movement (dH)</div>
                            <div class="stat-val" id="statDH">0.0000</div>
                        </div>
                    </div>
                </div>
            </details>

            <footer class="privacy-note">
                🔒 <b>Privacy Notice:</b> All image processing occurs in real-time on your computer. No photos, video recordings, or facial scans are ever saved, stored, or transmitted over the internet.
            </footer>
        </div>

        <script>
            const emotionColors = {
                'Happy': '#22c55e',
                'Neutral': '#38bdf8',
                'Surprise': '#eab308',
                'Sad': '#f97316',
                'Angry': '#ef4444',
                'Fear': '#a855f7',
                'Disgust': '#10b981'
            };

            const emotions = ['Happy', 'Neutral', 'Surprise', 'Sad', 'Angry', 'Fear', 'Disgust'];
            const container = document.getElementById('probContainer');

            emotions.forEach(emo => {
                const div = document.createElement('div');
                div.className = 'prob-row';
                div.innerHTML = `
                    <div class="prob-labels">
                        <span>${emo}</span>
                        <span id="p-val-${emo}">0%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" id="p-bar-${emo}" style="background-color: ${emotionColors[emo]}"></div>
                    </div>
                `;
                container.appendChild(div);
            });

            let isCameraActive = true;
            let supportDismissed = false;

            async function toggleCamera() {
                const endpoint = isCameraActive ? '/api/camera/stop' : '/api/camera/start';
                const res = await fetch(endpoint, { method: 'POST' });
                const data = await res.json();
                isCameraActive = !isCameraActive;

                const btn = document.getElementById('btnCameraToggle');
                const pill = document.getElementById('cameraPill');

                if (isCameraActive) {
                    btn.innerText = 'Pause Camera';
                    btn.classList.remove('btn-danger');
                    pill.innerText = '● Camera Active';
                    pill.className = 'pill pill-active';
                } else {
                    btn.innerText = 'Start Camera';
                    btn.classList.add('btn-danger');
                    pill.innerText = '○ Camera Paused';
                    pill.className = 'pill pill-paused';
                }
            }

            async function toggleEnhancer() {
                const res = await fetch('/api/toggle_enhancer', { method: 'POST' });
                const data = await res.json();
                document.getElementById('btnEnhancer').innerText = `Light Enhancer: ${data.enhancement_active ? 'ON' : 'OFF'}`;
            }

            async function resetUser() {
                await fetch('/api/reset_user', { method: 'POST' });
                document.getElementById('calibBanner').style.display = 'block';
                document.getElementById('calibBarFill').style.width = '0%';
            }

            async function sendFeeling(feeling) {
                try {
                    const res = await fetch('/api/user_feedback', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({feeling: feeling})
                    });
                    const data = await res.json();
                    const fb = document.getElementById('feelingFeedback');
                    fb.innerText = data.message;
                    fb.style.display = 'block';
                    setTimeout(() => { fb.style.display = 'none'; }, 6000);
                } catch(e) {
                    console.error("Feedback error:", e);
                }
            }

            function dismissSupport() {
                supportDismissed = true;
                document.getElementById('supportBanner').style.display = 'none';
            }

            // Screen reader debounce tracker
            let lastAnnouncedEmotion = "";

            async function updateTelemetry() {
                try {
                    const res = await fetch('/api/telemetry');
                    const data = await res.json();

                    // 1. Primary Expression Display
                    const heroIcon = document.getElementById('heroIcon');
                    const heroEmotion = document.getElementById('heroEmotion');
                    const heroConfidence = document.getElementById('heroConfidence');
                    const heroStatusMsg = document.getElementById('heroStatusMsg');

                    heroIcon.innerText = data.icon || '👤';
                    heroEmotion.innerText = data.display_label || 'Checking...';
                    heroConfidence.innerText = `Confidence: ${data.confidence_level || 'Moderate'} (${Math.round(data.confidence)}%)`;
                    heroStatusMsg.innerText = data.status_message || '';

                    // Soft color based on emotion / confidence
                    if (data.is_confident && emotionColors[data.primary_emotion]) {
                        heroEmotion.style.color = emotionColors[data.primary_emotion];
                    } else {
                        heroEmotion.style.color = '#cbd5e1';
                    }

                    // 2. Guidance & Environmental Indicators
                    const guidancePill = document.getElementById('guidancePill');
                    const guidanceText = document.getElementById('envGuidanceText');

                    if (data.guidance_message) {
                        guidanceText.innerText = data.guidance_message;
                    }

                    const brightness = data.quality_metrics ? data.quality_metrics.brightness : 100;
                    if (brightness < 30) {
                        guidancePill.innerText = '💡 Low Light';
                        guidancePill.style.color = '#f59e0b';
                    } else {
                        guidancePill.innerText = '☀️ Lighting Good';
                        guidancePill.style.color = '#93c5fd';
                    }

                    // 3. Calibration banner
                    const calibBanner = document.getElementById('calibBanner');
                    const calibFill = document.getElementById('calibBarFill');
                    if (data.baseline_calibrated) {
                        calibBanner.style.display = 'none';
                    } else {
                        calibBanner.style.display = 'block';
                        calibFill.style.width = `${Math.round((data.calibration_progress || 0) * 100)}%`;
                    }

                    // 4. Calm Wellbeing Support Check-In (Non-alarmist)
                    const supportBanner = document.getElementById('supportBanner');
                    if (data.support_checkin && !supportDismissed) {
                        supportBanner.style.display = 'block';
                        if (data.support_message) {
                            document.getElementById('supportText').innerText = data.support_message;
                        }
                    } else {
                        supportBanner.style.display = 'none';
                    }

                    // 5. Technical Distribution
                    emotions.forEach(emo => {
                        const pct = data.probabilities[emo] || 0;
                        const elVal = document.getElementById(`p-val-${emo}`);
                        const elBar = document.getElementById(`p-bar-${emo}`);
                        if (elVal) elVal.innerText = `${pct}%`;
                        if (elBar) elBar.style.width = `${pct}%`;
                    });

                    document.getElementById('statFps').innerText = data.fps || 0.0;
                    document.getElementById('statLight').innerText = Math.round(brightness);
                    document.getElementById('statSharp').innerText = Math.round(data.quality_metrics ? data.quality_metrics.sharpness : 0);

                    if (data.geo_debug && data.geo_debug.geo_active) {
                        const dw = data.geo_debug.dW !== null ? (data.geo_debug.dW > 0 ? '+' : '') + data.geo_debug.dW.toFixed(4) : '0.0000';
                        const dh = data.geo_debug.dH !== null ? (data.geo_debug.dH > 0 ? '+' : '') + data.geo_debug.dH.toFixed(4) : '0.0000';
                        document.getElementById('statDW').innerText = dw;
                        document.getElementById('statDH').innerText = dh;
                    }

                } catch (e) {
                    console.error("Telemetry error:", e);
                }
            }

            setInterval(updateTelemetry, 250);
        </script>
    </body>
    </html>
    """

def open_browser_delayed():
    time.sleep(1.2)
    print(" [DOSTH AI] Opening web browser at: http://localhost:8000")
    webbrowser.open("http://localhost:8000")

if __name__ == '__main__':
    print("==========================================================")
    print("   DOSTH AI - Elderly-Accessible Expression Assistant    ")
    print("==========================================================")
    print(" Serving at: http://localhost:8000")
    print(" Press Ctrl + C to stop.")
    print("==========================================================\n")
    
    threading.Thread(target=open_browser_delayed, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
