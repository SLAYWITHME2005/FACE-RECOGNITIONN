"""
DOSTH AI — Data Collection Script for Elderly Emotion Dataset
==============================================================
Collects labeled face images from the webcam to build a Malayali elderly
emotion dataset for fine-tuning the FER model.

Usage:
    python scripts/collect_training_data.py

Controls:
    1  →  Save frame labeled: Angry
    2  →  Save frame labeled: Disgust
    3  →  Save frame labeled: Fear
    4  →  Save frame labeled: Happy
    5  →  Save frame labeled: Sad
    6  →  Save frame labeled: Surprise
    7  →  Save frame labeled: Neutral
    q  →  Quit and show collection summary

Target: Collect at least 30-50 samples per emotion for best results.
        More samples = better accuracy for elderly faces.

Dataset Recommendation:
    Supplement webcam data with:
    • FACES Dataset (Max Planck) — Ages 58-86
      URL: https://faces.mpib-berlin.mpg.de/ger/start.html
    • RAF-DB — Real-world webcam quality
      URL: http://whdeng.cn/RAF/model1.html
"""

import cv2
import os
import sys
import time
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.fer.face_detector import FaceDetector
from modules.fer.image_enhancer import ImageEnhancer

EMOTIONS = {
    '1': 'Angry',
    '2': 'Disgust',
    '3': 'Fear',
    '4': 'Happy',
    '5': 'Sad',
    '6': 'Surprise',
    '7': 'Neutral'
}

EMOTION_COLORS = {
    'Angry':    (40,  40,  240),
    'Disgust':  (50,  140, 90),
    'Fear':     (200, 50,  180),
    'Happy':    (0,   220, 100),
    'Sad':      (255, 140, 0),
    'Surprise': (0,   215, 255),
    'Neutral':  (200, 200, 200)
}


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'collected', 'elderly_faces')

    for emotion in EMOTIONS.values():
        os.makedirs(os.path.join(data_dir, emotion), exist_ok=True)

    detector = FaceDetector()
    enhancer = ImageEnhancer()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    counts = defaultdict(int)
    # Load existing counts
    for emotion in EMOTIONS.values():
        folder = os.path.join(data_dir, emotion)
        counts[emotion] = len([f for f in os.listdir(folder) if f.endswith('.jpg')])

    last_saved_emotion = None
    last_saved_time = 0
    flash_duration = 0.5  # seconds of green flash on save

    print("\n" + "=" * 60)
    print("  DOSTH AI — Elderly Emotion Data Collector")
    print("=" * 60)
    print("  Press number keys to SAVE a labeled face frame:")
    for key, emotion in EMOTIONS.items():
        print(f"    [{key}]  →  {emotion}")
    print("  [Q]  →  Quit and show summary")
    print("=" * 60)
    print("  Tip: Make exaggerated expressions so the camera can capture them.")
    print("  Target: 30-50 images per emotion for good fine-tuning results.")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = frame.shape[:2]

        primary_face = detector.get_primary_face(frame)
        face_detected = primary_face is not None

        # Draw face box
        if face_detected:
            x, y, fw, fh = primary_face['bbox']
            cv2.rectangle(display, (x, y), (x + fw, y + fh), (0, 255, 120), 2)
            c = int(min(fw, fh) * 0.2)
            for pts in [((x, y), (x + c, y)), ((x, y), (x, y + c)),
                        ((x + fw, y), (x + fw - c, y)), ((x + fw, y), (x + fw, y + c)),
                        ((x, y + fh), (x + c, y + fh)), ((x, y + fh), (x, y + fh - c)),
                        ((x + fw, y + fh), (x + fw - c, y + fh)), ((x + fw, y + fh), (x + fw, y + fh - c))]:
                cv2.line(display, pts[0], pts[1], (0, 255, 120), 3)

        # Flash effect after saving
        now = time.time()
        if last_saved_emotion and (now - last_saved_time) < flash_duration:
            flash_color = EMOTION_COLORS.get(last_saved_emotion, (0, 255, 0))
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), flash_color, -1)
            display = cv2.addWeighted(display, 0.75, overlay, 0.25, 0)
            cv2.putText(display, f"SAVED: {last_saved_emotion}!",
                        (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)

        # Instruction panel
        panel_x = w - 260
        cv2.rectangle(display, (panel_x, 0), (w, h), (15, 20, 35), -1)
        cv2.putText(display, "DOSTH AI", (panel_x + 10, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 180, 255), 2)
        cv2.putText(display, "Data Collector", (panel_x + 10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 180), 1)

        cv2.line(display, (panel_x + 10, 68), (w - 10, 68), (40, 50, 80), 1)

        y_off = 95
        for key, emotion in EMOTIONS.items():
            count = counts[emotion]
            color = (0, 220, 100) if count >= 30 else (200, 200, 200)
            bar_pct = min(1.0, count / 30.0)
            bar_w = int(220 * bar_pct)
            cv2.rectangle(display, (panel_x + 10, y_off + 14),
                          (panel_x + 10 + bar_w, y_off + 22), color, -1)
            cv2.putText(display, f"[{key}] {emotion}: {count}",
                        (panel_x + 10, y_off + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            y_off += 38

        status_color = (0, 220, 100) if face_detected else (0, 80, 220)
        status_text = "FACE DETECTED" if face_detected else "NO FACE"
        cv2.putText(display, status_text, (panel_x + 10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(display, "[Q] Quit & Save Summary",
                    (panel_x + 10, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow('DOSTH AI - Data Collector', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        key_char = chr(key) if key != 255 else ''
        if key_char in EMOTIONS and face_detected:
            emotion = EMOTIONS[key_char]
            face_crop = primary_face['cropped_face']
            enhanced_face, _ = enhancer.enhance(face_crop)
            resized = cv2.resize(enhanced_face, (128, 128))

            timestamp = int(time.time() * 1000)
            save_path = os.path.join(data_dir, emotion, f"{emotion}_{timestamp}.jpg")
            cv2.imwrite(save_path, resized)
            counts[emotion] += 1
            last_saved_emotion = emotion
            last_saved_time = time.time()
            print(f"[SAVED] {emotion} → {save_path}  (total: {counts[emotion]})")

    cap.release()
    cv2.destroyAllWindows()

    # Save summary JSON
    summary_path = os.path.join(data_dir, 'collection_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(dict(counts), f, indent=2)

    print("\n" + "=" * 60)
    print("  Collection Complete! Summary:")
    print("=" * 60)
    total = 0
    for emotion, count in sorted(counts.items()):
        status = "✓ Ready" if count >= 30 else f"⚠ Need {30 - count} more"
        print(f"  {emotion:12s}: {count:3d} images  [{status}]")
        total += count
    print(f"\n  Total collected: {total} images")
    print(f"  Saved to: {data_dir}")
    print("\n  Next step → Run: python scripts/finetune_elderly.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
