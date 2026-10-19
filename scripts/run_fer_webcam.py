import cv2
import time
import numpy as np
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fer.pipeline import FacialEmotionPipeline

EMOTION_COLORS = {
    'Happy':    (0, 220, 100),   # Vibrant Green
    'Neutral':  (230, 210, 160), # Soft Slate/Cyan
    'Surprise': (0, 215, 255),   # Golden Yellow
    'Sad':      (255, 140, 0),   # Deep Blue-Orange
    'Fear':     (200, 50, 180),  # Purple
    'Angry':    (40, 40, 240),   # Strong Red
    'Disgust':  (50, 140, 90)    # Forest Green
}

def draw_hud(frame: np.ndarray, result: dict, fps: float, enable_enhancement: bool) -> np.ndarray:
    """Renders a sleek presentation overlay for DOSTH AI reviews and demos."""
    h, w = frame.shape[:2]
    
    # 1. Draw Bounding Box and Emotion Tag if face detected
    if result['face_detected']:
        x, y, fw, fh = result['bounding_box']
        emotion = result['primary_emotion']
        conf = result['confidence']
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        
        # High-visibility Bounding Box
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 3)
        corner_len = int(min(fw, fh) * 0.22)
        cv2.line(frame, (x, y), (x + corner_len, y), color, 5)
        cv2.line(frame, (x, y), (x, y + corner_len), color, 5)
        cv2.line(frame, (x + fw, y), (x + fw - corner_len, y), color, 5)
        cv2.line(frame, (x + fw, y), (x + fw, y + corner_len), color, 5)
        cv2.line(frame, (x, y + fh), (x + corner_len, y + fh), color, 5)
        cv2.line(frame, (x, y + fh), (x, y + fh - corner_len), color, 5)
        cv2.line(frame, (x + fw, y + fh), (x + fw - corner_len, y + fh), color, 5)
        cv2.line(frame, (x + fw, y + fh), (x + fw, y + fh - corner_len), color, 5)
        
        # Prominent Floating Badge
        label_text = f" {emotion.upper()} : {int(conf * 100)}% "
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        bg_y1 = max(0, y - th - 16)
        cv2.rectangle(frame, (x, bg_y1), (x + tw + 10, y), color, -1)
        cv2.putText(frame, label_text, (x + 5, y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)

    # 2. Right Telemetry Sidebar
    sidebar_w = 280
    sidebar_x = w - sidebar_w - 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (sidebar_x, 15), (w - 15, h - 15), (15, 18, 25), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (sidebar_x, 15), (w - 15, h - 15), (70, 85, 105), 2)

    # Sidebar Header
    cv2.putText(frame, "DOSTH AI", (sidebar_x + 15, 48), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Multimodal Assistant", (sidebar_x + 15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1)
    
    status_color = (0, 255, 120) if enable_enhancement else (160, 160, 160)
    cv2.putText(frame, f"Enhance: {'ON' if enable_enhancement else 'OFF'}", (sidebar_x + 15, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (sidebar_x + 175, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    
    cv2.line(frame, (sidebar_x + 15, 105), (w - 25, 105), (80, 95, 115), 1)

    # Live Probability Distribution Bars
    start_y = 138
    probs = result['probabilities']
    for idx, (emo, p_val) in enumerate(probs.items()):
        y_pos = start_y + (idx * 34)
        cv2.putText(frame, f"{emo[:5]}", (sidebar_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
        
        # Track
        bar_x = sidebar_x + 75
        bar_max_w = 125
        cv2.rectangle(frame, (bar_x, y_pos - 12), (bar_x + bar_max_w, y_pos + 2), (40, 45, 55), -1)
        
        # Fill
        bar_fill = int(bar_max_w * p_val)
        b_color = EMOTION_COLORS.get(emo, (200, 200, 200))
        if bar_fill > 0:
            cv2.rectangle(frame, (bar_x, y_pos - 12), (bar_x + bar_fill, y_pos + 2), b_color, -1)
            
        cv2.putText(frame, f"{int(p_val * 100)}%", (bar_x + bar_max_w + 8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    # Lighting / Clarity Telemetry
    q = result.get('quality_metrics') or {'brightness': 0, 'sharpness': 0}
    q_y = start_y + (7 * 34) + 18
    cv2.line(frame, (sidebar_x + 15, q_y - 12), (w - 25, q_y - 12), (80, 95, 115), 1)
    cv2.putText(frame, f"Light Level: {q.get('brightness', 0):.0f}", (sidebar_x + 15, q_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1)
    cv2.putText(frame, f"Focus Clarity: {q.get('sharpness', 0):.0f}", (sidebar_x + 15, q_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1)

    # Sustained Distress Caregiver Alert Banner
    if result.get('distress_alert', False):
        cv2.rectangle(frame, (25, 25), (380, 85), (0, 0, 220), -1)
        cv2.putText(frame, "! CAREGIVER ALERT !", (40, 55), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Sustained distress/sadness detected", (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # Bottom helper
    cv2.putText(frame, "[E] Toggle Enhancer  |  [S] Snapshot  |  [Q / ESC / Click X] Exit", (25, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return frame

def main():
    print("====================================================")
    print("      DOSTH AI - Live Facial Emotion Recognition    ")
    print("====================================================")
    
    print("[1/3] Loading AI Models...")
    pipeline = FacialEmotionPipeline()
    
    print("[2/3] Connecting to Camera (DirectShow)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[WARN] DirectShow failed, trying default camera index...")
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("[ERROR] Camera not found or in use by another app. Please close other camera apps and retry.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    window_name = "DOSTH AI - Emotion-Aware Assistant"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, 680)
    # Bring window to front
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    enable_enhancement = True
    prev_time = time.time()
    fps = 0.0
    last_terminal_log = time.time()

    print("[3/3] Window is now OPEN on your screen!")
    print("----------------------------------------------------")
    print(" CONTROLS:")
    print("   * Close Window: Click the [X] button or press 'q' / 'ESC'")
    print("   * Stop Terminal: Press Ctrl + C")
    print("   * Enhancer: Press 'e'")
    print("   * Snapshot: Press 's'")
    print("----------------------------------------------------\n")
    
    try:
        while True:
            # Check if user closed the window with the mouse [X] button
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("[DOSTH AI] Window closed with [X] button. Exiting cleanly...")
                    break
            except Exception:
                break

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
                
            frame = cv2.flip(frame, 1) # Mirror for natural presentation
            
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
            prev_time = curr_time
            
            # Process frame through FER pipeline
            result = pipeline.process_frame(frame, enable_enhancement=enable_enhancement)
            
            # Log to terminal console periodically
            if curr_time - last_terminal_log >= 1.5:
                if result['face_detected']:
                    print(f"[LIVE] Emotion: {result['primary_emotion']:<10} | Confidence: {result['confidence']*100:.1f}% | Valence: {result['valence']:+.1f} | FPS: {fps:.1f}")
                else:
                    print(f"[LIVE] Detecting face... (FPS: {fps:.1f})")
                last_terminal_log = curr_time
            
            # Render HUD overlay on video frame
            display_frame = draw_hud(frame, result, fps, enable_enhancement)
            
            cv2.imshow(window_name, display_frame)
            
            # Wait for key event (checks 'q', 'Q', ESC)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                print("[DOSTH AI] Exit key pressed. Closing...")
                break
            elif key == ord('e') or key == ord('E'):
                enable_enhancement = not enable_enhancement
                print(f"[DOSTH AI] Enhancement {'Enabled' if enable_enhancement else 'Disabled'}")
            elif key == ord('s') or key == ord('S'):
                snap_path = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_path, display_frame)
                print(f"[DOSTH AI] Saved snapshot: {snap_path}")
                
    except KeyboardInterrupt:
        print("\n[DOSTH AI] Stopped via terminal Ctrl+C.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Force flush windows event loop on Windows
        for _ in range(5):
            cv2.waitKey(1)
        print("[DOSTH AI] Clean shutdown complete.")

if __name__ == '__main__':
    main()
