import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.fer.pipeline import FacialEmotionPipeline

def test_camera_and_pipeline():
    print("[1] Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Camera 0 could not be opened! Check camera permissions or if another app is using it.")
        return False
    
    print("[2] Camera opened successfully! Initializing pipeline...")
    pipeline = FacialEmotionPipeline()

    print("[3] Capturing 10 test frames...")
    faces_found = 0
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"  Frame {i+1}: Failed to grab frame.")
            continue
        
        result = pipeline.process_frame(frame)
        if result['face_detected']:
            faces_found += 1
            print(f"  Frame {i+1}: Face DETECTED! Emotion={result['primary_emotion']} ({result['confidence']*100:.1f}%), Geo={result.get('geo_debug')}")
        else:
            print(f"  Frame {i+1}: No face in view (normal if camera is uncovered or person is not in front)")

    cap.release()
    print(f"\n[SUMMARY] Camera works! {faces_found}/10 frames had face detected.")
    return True

if __name__ == "__main__":
    test_camera_and_pipeline()
