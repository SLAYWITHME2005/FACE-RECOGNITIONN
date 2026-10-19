"""
DOSTH AI — 5 Core Emotions Accuracy Test
=========================================
Tests the accuracy and precision of the 5 target emotions:
  1. Happy
  2. Neutral
  3. Sad
  4. Fear
  5. Angry

Verifies that the hybrid neuro-geometric affective pipeline correctly
identifies and differentiates each emotion state.
"""

import unittest
import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fer.emotion_model import EmotionClassifier
from modules.fer.pipeline import FacialEmotionPipeline


class Test5EmotionsAccuracy(unittest.TestCase):

    def setUp(self):
        self.classifier = EmotionClassifier()
        # Calibrate a standard resting baseline
        dummy_face = np.full((112, 112, 3), 128, dtype=np.uint8)
        neutral_landmarks = {
            'right_eye': (38, 42),
            'left_eye': (74, 42),
            'nose': (56, 62),
            'right_mouth': (40, 84),
            'left_mouth': (72, 84)
        }
        for _ in range(40):
            self.classifier.collect_baseline(dummy_face, landmarks=neutral_landmarks)

    def test_neutral_accuracy(self):
        """Resting face with relaxed muscles should classify as Neutral."""
        face = np.full((112, 112, 3), 128, dtype=np.uint8)
        neutral_landmarks = {
            'right_eye': (38, 42),
            'left_eye': (74, 42),
            'nose': (56, 62),
            'right_mouth': (40, 84),
            'left_mouth': (72, 84)
        }
        pred = self.classifier.predict(face, landmarks=neutral_landmarks)
        self.assertEqual(pred['primary_emotion'], 'Neutral')
        self.assertGreater(pred['probabilities']['Neutral'], 0.40)
        print(f"\n[TEST PASSED] NEUTRAL -> Conf: {pred['confidence']:.1%}, Probs: {pred['probabilities']}")

    def test_happy_accuracy(self):
        """Smiling face (mouth widened, corners upturned) should classify as Happy."""
        face = np.full((112, 112, 3), 140, dtype=np.uint8)
        # Widened mouth (dx = 42 instead of 32)
        happy_landmarks = {
            'right_eye': (38, 42),
            'left_eye': (74, 42),
            'nose': (56, 62),
            'right_mouth': (35, 80),
            'left_mouth': (77, 80)
        }
        pred = self.classifier.predict(face, landmarks=happy_landmarks)
        self.assertEqual(pred['primary_emotion'], 'Happy')
        self.assertGreater(pred['probabilities']['Happy'], 0.50)
        print(f"[TEST PASSED] HAPPY   -> Conf: {pred['confidence']:.1%}, Probs: {pred['probabilities']}")

    def test_sad_accuracy(self):
        """Frowning face (mouth corners drooping down away from nose) should classify as Sad."""
        face = np.full((112, 112, 3), 110, dtype=np.uint8)
        # Drooping mouth corners (y = 92 instead of 84)
        sad_landmarks = {
            'right_eye': (38, 42),
            'left_eye': (74, 42),
            'nose': (56, 60),
            'right_mouth': (41, 92),
            'left_mouth': (71, 92)
        }
        pred = self.classifier.predict(face, landmarks=sad_landmarks)
        self.assertEqual(pred['primary_emotion'], 'Sad')
        self.assertGreater(pred['probabilities']['Sad'], 0.40)
        print(f"[TEST PASSED] SAD     -> Conf: {pred['confidence']:.1%}, Probs: {pred['probabilities']}")

    def test_angry_accuracy(self):
        """Scowling/compressed face (pursed, tightened mouth) should classify as Angry."""
        face = np.full((112, 112, 3), 120, dtype=np.uint8)
        # Narrowed/tightened lips (width = 24 instead of 32)
        angry_landmarks = {
            'right_eye': (38, 44),
            'left_eye': (74, 44),
            'nose': (56, 62),
            'right_mouth': (45, 84),
            'left_mouth': (67, 84)
        }
        pred = self.classifier.predict(face, landmarks=angry_landmarks)
        self.assertEqual(pred['primary_emotion'], 'Angry')
        self.assertGreater(pred['probabilities']['Angry'], 0.35)
        print(f"[TEST PASSED] ANGRY   -> Conf: {pred['confidence']:.1%}, Probs: {pred['probabilities']}")

    def test_fear_accuracy(self):
        """Fearful expression (open/elongated mouth) should classify as Fear."""
        face = np.full((112, 112, 3), 125, dtype=np.uint8)
        # Elongated mouth/jaw drop with widened mouth
        fear_landmarks = {
            'right_eye': (38, 40),
            'left_eye': (74, 40),
            'nose': (56, 60),
            'right_mouth': (35, 94),
            'left_mouth': (77, 94)
        }
        pred = self.classifier.predict(face, landmarks=fear_landmarks)
        self.assertEqual(pred['primary_emotion'], 'Fear')
        self.assertGreater(pred['probabilities']['Fear'], 0.35)
        print(f"[TEST PASSED] FEAR    -> Conf: {pred['confidence']:.1%}, Probs: {pred['probabilities']}")


if __name__ == '__main__':
    unittest.main()
