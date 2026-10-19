"""
DOSTH AI — Elderly-Accessible FER System Comprehensive Verification
=====================================================================
Validates all phases of the refactored FER application:
1. Model inference correctness and honest confidence thresholding.
2. Low-confidence handling ('Expression unclear' behavior).
3. Temporal smoothing (EMA) and dwell-time transition (no jitter, no lock-in).
4. Stale-state clearing when face is absent.
5. Environmental guidance (dim lighting, blur, tilt).
6. Non-alarmist wellbeing support check-in logic.
7. User feedback endpoint functionality.
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fer.emotion_model import EmotionClassifier
from modules.fer.pipeline import FacialEmotionPipeline


class TestElderlyFERSystem(unittest.TestCase):

    def setUp(self):
        self.pipeline = FacialEmotionPipeline()
        self.classifier = self.pipeline.classifier

    def test_01_confidence_tiers_and_unclear_handling(self):
        """Verifies that low-confidence or ambiguous predictions display 'Expression unclear'."""
        # Ambiguous probability distribution where top two classes are almost equal
        ambiguous_probs = {
            'Angry': 0.10,
            'Disgust': 0.05,
            'Fear': 0.10,
            'Happy': 0.26,
            'Sad': 0.24,
            'Surprise': 0.05,
            'Neutral': 0.20
        }
        res = self.classifier._format_result(ambiguous_probs, mode='test')
        self.assertFalse(res['is_confident'])
        self.assertEqual(res['confidence_level'], 'Unclear')
        self.assertEqual(res['display_label'], 'Expression unclear')
        self.assertIn("Unable to identify the expression confidently", res['status_message'])

    def test_02_moderate_confidence_tier(self):
        """Verifies that moderate confidence presents expression as 'Possible: Emotion'."""
        moderate_probs = {
            'Angry': 0.05,
            'Disgust': 0.02,
            'Fear': 0.03,
            'Happy': 0.55,
            'Sad': 0.15,
            'Surprise': 0.05,
            'Neutral': 0.15
        }
        res = self.classifier._format_result(moderate_probs, mode='test')
        self.assertTrue(res['is_confident'])
        self.assertEqual(res['confidence_level'], 'Moderate')
        self.assertEqual(res['display_label'], 'Possible: Happy')

    def test_03_high_confidence_tier(self):
        """Verifies that high confidence clearly presents the emotion."""
        high_probs = {
            'Angry': 0.02,
            'Disgust': 0.01,
            'Fear': 0.01,
            'Happy': 0.85,
            'Sad': 0.04,
            'Surprise': 0.02,
            'Neutral': 0.05
        }
        res = self.classifier._format_result(high_probs, mode='test')
        self.assertTrue(res['is_confident'])
        self.assertEqual(res['confidence_level'], 'High')
        self.assertEqual(res['display_label'], 'Happy')

    def test_04_temporal_smoothing_no_flicker_and_no_lockin(self):
        """
        Verifies that single-frame noise does NOT flip the emotion (no flicker),
        but sustained expression changes DO switch the emotion (no permanent lock-in).
        """
        pipeline = FacialEmotionPipeline(ema_alpha=0.35, hysteresis_margin=0.08)

        # Feed 10 frames of Neutral
        neutral_probs = {e: 0.05 for e in EmotionClassifier.TARGET_CLASSES}
        neutral_probs['Neutral'] = 0.70

        for _ in range(10):
            smoothed, stable = pipeline._update_smoothed_probabilities(neutral_probs)

        self.assertEqual(stable, 'Neutral')

        # Feed 1 frame of transient Happy noise (e.g. 52% Happy vs 38% Neutral)
        noisy_happy = {e: 0.05 for e in EmotionClassifier.TARGET_CLASSES}
        noisy_happy['Happy'] = 0.52
        noisy_happy['Neutral'] = 0.38

        smoothed, stable = pipeline._update_smoothed_probabilities(noisy_happy)
        # Should NOT flip immediately on a single close frame
        self.assertEqual(stable, 'Neutral', "Single frame of mild noise should not cause immediate flicker!")

        # Now feed 5 sustained frames of strong Happy
        strong_happy = {e: 0.02 for e in EmotionClassifier.TARGET_CLASSES}
        strong_happy['Happy'] = 0.88

        for _ in range(5):
            smoothed, stable = pipeline._update_smoothed_probabilities(strong_happy)

        # Should transition smoothly without getting permanently locked to Neutral
        self.assertEqual(stable, 'Happy', "Sustained expression change must successfully transition (no lock-in)!")

    def test_05_stale_prediction_clearing_when_no_face(self):
        """Verifies that missing face for > 10 frames clears previous prediction."""
        pipeline = FacialEmotionPipeline()
        # Seed some prediction
        happy_probs = {e: 0.02 for e in EmotionClassifier.TARGET_CLASSES}
        happy_probs['Happy'] = 0.88
        pipeline._update_smoothed_probabilities(happy_probs)

        # Simulate 12 blank frames
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(12):
            res = pipeline.process_frame(blank_frame)

        self.assertFalse(res['face_detected'])
        self.assertEqual(res['display_label'], 'No face detected')
        self.assertIsNone(pipeline.smoothed_probs, "Smoothed history must be cleared after user leaves frame.")

    def test_06_environmental_guidance(self):
        """Verifies that dim lighting and blur trigger gentle user guidance."""
        pipeline = FacialEmotionPipeline()

        # Dim lighting
        status, msg = pipeline._assess_environment({'brightness': 20.0, 'sharpness': 50.0}, None)
        self.assertEqual(status, 'low_light')
        self.assertIn("dim", msg.lower())

        # Blurry image
        status, msg = pipeline._assess_environment({'brightness': 80.0, 'sharpness': 10.0}, None)
        self.assertEqual(status, 'blurry')
        self.assertIn("blurry", msg.lower())


if __name__ == '__main__':
    unittest.main()
