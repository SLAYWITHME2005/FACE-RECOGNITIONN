import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fer.face_detector import FaceDetector
from modules.fer.image_enhancer import ImageEnhancer
from modules.fer.emotion_model import EmotionClassifier
from modules.fer.pipeline import FacialEmotionPipeline

class TestFERPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = FacialEmotionPipeline()
        self.enhancer = ImageEnhancer()
        self.classifier = EmotionClassifier()

    def test_image_enhancer_low_light(self):
        # Create dark image
        dark_img = np.full((100, 100, 3), 20, dtype=np.uint8)
        enhanced, metrics = self.enhancer.enhance(dark_img)
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, dark_img.shape)
        # Enhanced brightness should be higher than initial 20
        self.assertGreater(metrics['brightness'], 20.0)

    def test_emotion_classifier_shape(self):
        dummy_face = np.full((100, 100, 3), 128, dtype=np.uint8)
        pred = self.classifier.predict(dummy_face)
        self.assertIn('primary_emotion', pred)
        self.assertIn('confidence', pred)
        self.assertIn('probabilities', pred)
        self.assertEqual(len(pred['probabilities']), 7)
        # Probabilities should sum to approximately 1.0
        total_p = sum(pred['probabilities'].values())
        self.assertAlmostEqual(total_p, 1.0, places=2)

    def test_pipeline_blank_frame(self):
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.pipeline.process_frame(blank_frame)
        self.assertFalse(res['face_detected'])
        self.assertIsNone(res['bounding_box'])

    def test_temporal_smoothing(self):
        self.pipeline.reset_history()
        dummy_face = np.full((64, 64, 3), 150, dtype=np.uint8)
        
        # Simulate feeding 5 consecutive frames
        for _ in range(5):
            res = self.pipeline.process_frame(dummy_face)
            self.assertIn('primary_emotion', res)
            self.assertIn('confidence', res)

if __name__ == '__main__':
    unittest.main()
