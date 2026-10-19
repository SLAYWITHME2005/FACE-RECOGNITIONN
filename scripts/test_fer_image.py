import cv2
import numpy as np
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fer.pipeline import FacialEmotionPipeline

def create_synthetic_face_image() -> np.ndarray:
    """Creates a synthetic human-like face image for automated testing."""
    img = np.full((400, 400, 3), 40, dtype=np.uint8) # Dark room background
    
    # Draw face oval
    cv2.ellipse(img, (200, 200), (90, 120), 0, 0, 360, (180, 200, 230), -1)
    
    # Eyes
    cv2.circle(img, (165, 175), 14, (255, 255, 255), -1)
    cv2.circle(img, (165, 175), 6, (50, 40, 30), -1)
    cv2.circle(img, (235, 175), 14, (255, 255, 255), -1)
    cv2.circle(img, (235, 175), 6, (50, 40, 30), -1)
    
    # Eyebrows
    cv2.line(img, (150, 155), (180, 155), (40, 30, 20), 3)
    cv2.line(img, (220, 155), (250, 155), (40, 30, 20), 3)
    
    # Nose
    cv2.line(img, (200, 185), (200, 225), (140, 160, 190), 2)
    
    # Smile
    cv2.ellipse(img, (200, 250), (35, 18), 0, 0, 180, (50, 40, 120), 3)
    return img

def main():
    parser = argparse.ArgumentParser(description="DOSTH AI - Test FER Pipeline on Static Image")
    parser.add_argument('--image', type=str, default=None, help="Path to input image")
    parser.add_argument('--output', type=str, default=None, help="Path to save annotated result image")
    args = parser.parse_args()

    pipeline = FacialEmotionPipeline()

    if args.image and os.path.exists(args.image):
        print(f"[DOSTH AI] Loading test image: {args.image}")
        image = cv2.imread(args.image)
    else:
        print("[DOSTH AI] No image specified. Generating synthetic face image for pipeline verification...")
        image = create_synthetic_face_image()

    # Run processing
    result = pipeline.process_frame(image, enable_enhancement=True)
    
    # Create serializable output (excluding numpy arrays)
    clean_res = {
        'face_detected': result['face_detected'],
        'primary_emotion': result['primary_emotion'],
        'confidence': result['confidence'],
        'probabilities': result['probabilities'],
        'valence': result['valence'],
        'arousal': result['arousal'],
        'emotion_type': result['emotion_type'],
        'bounding_box': result['bounding_box'],
        'quality_metrics': result['quality_metrics'],
        'distress_alert': result['distress_alert']
    }

    print("\n--- FER PIPELINE RESULTS ---")
    print(json.dumps(clean_res, indent=2))
    print("----------------------------\n")
    
    if args.output:
        if result['face_detected']:
            x, y, w, h = result['bounding_box']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"{result['primary_emotion']} ({int(result['confidence']*100)}%)", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(args.output, image)
        print(f"[DOSTH AI] Saved annotated output image to {args.output}")

if __name__ == '__main__':
    main()

