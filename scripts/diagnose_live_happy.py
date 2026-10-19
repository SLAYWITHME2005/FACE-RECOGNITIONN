import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.fer.pipeline import FacialEmotionPipeline

def diagnose():
    print("Initializing pipeline...")
    pipeline = FacialEmotionPipeline()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    print("Reading 20 frames from camera...")
    for i in range(20):
        ret, frame = cap.read()
        if not ret:
            continue
        res = pipeline.process_frame(frame)
        det = res.get('face_detected')
        emo = res.get('primary_emotion')
        conf = res.get('confidence')
        probs = res.get('probabilities')
        geo = res.get('geo_debug', {})
        print(f"Frame {i+1:02d}: face={det}, emo={emo}, conf={conf:.2f}, dW={geo.get('dW')}, dH={geo.get('dH')}, resting={geo.get('is_resting')}")
        if det:
            print(f"   Probs: {probs}")

    cap.release()

if __name__ == '__main__':
    diagnose()
