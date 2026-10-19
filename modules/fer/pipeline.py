import cv2
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple

from .face_detector import FaceDetector
from .image_enhancer import ImageEnhancer
from .emotion_model import EmotionClassifier

class FacialEmotionPipeline:
    """
    Unified Real-Time Facial Expression Pipeline for DOSTH AI Elderly Care.

    Key Reliability & Stability Features:
    1. Confidence-Aware Temporal Smoothing (EMA with momentum, no flicker, no lock-in).
    2. Hysteresis Protection: Requires sustained emotion signals to switch labels, preventing frame jitter.
    3. Stale State Clearing: Clears previous predictions when the user leaves the camera.
    4. Real-Time Environmental Guidance: Gently detects low lighting, blur, tilt, or distance.
    5. Non-Alarmist Wellbeing Check-in: Replaces alarming red banners with a calm, supportive prompt.
    """

    SUPPORT_CHECKIN_EMOTIONS = ('Sad', 'Fear')

    def __init__(self,
                 model_path: Optional[str] = None,
                 ema_alpha: float = 0.35,
                 hysteresis_margin: float = 0.08):
        """
        Args:
            ema_alpha: Smoothing weight for new frame probabilities (0.35 = balanced responsiveness & stability).
            hysteresis_margin: Extra margin a candidate emotion needs to dethrone current emotion instantly.
        """
        self.detector = FaceDetector()
        self.enhancer = ImageEnhancer()
        self.classifier = EmotionClassifier(model_path=model_path)

        self.ema_alpha = ema_alpha
        self.hysteresis_margin = hysteresis_margin

        # Smoothed state
        self.smoothed_probs: Optional[Dict[str, float]] = None
        self.current_stable_emotion: str = "Neutral"
        self.candidate_emotion: Optional[str] = None
        self.candidate_consecutive_count: int = 0

        # Face tracking & stale state counters
        self.missing_face_count: int = 0
        self.consecutive_support_frames: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public Controls
    # ─────────────────────────────────────────────────────────────────────────

    def reset_for_new_user(self):
        """Resets baseline calibration and smoothing when a new person sits down."""
        self.smoothed_probs = None
        self.current_stable_emotion = "Neutral"
        self.candidate_emotion = None
        self.candidate_consecutive_count = 0
        self.missing_face_count = 0
        self.consecutive_support_frames = 0
        self.classifier.reset_baseline()

    def reset_history(self):
        """Resets temporal smoothing buffers while keeping resting calibration."""
        self.smoothed_probs = None
        self.current_stable_emotion = "Neutral"
        self.candidate_emotion = None
        self.candidate_consecutive_count = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal Smoothing & Hysteresis
    # ─────────────────────────────────────────────────────────────────────────

    def _update_smoothed_probabilities(self, new_probs: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """
        Applies Exponential Moving Average (EMA) and hysteresis dwell protection.
        Prevents rapid flickering between emotions on adjacent frames without
        permanently locking to one emotion.
        """
        if self.smoothed_probs is None:
            self.smoothed_probs = new_probs.copy()
            self.current_stable_emotion = max(new_probs, key=new_probs.get)
            return self.smoothed_probs, self.current_stable_emotion

        # 1. Update EMA probabilities
        for emo in EmotionClassifier.TARGET_CLASSES:
            old_p = self.smoothed_probs.get(emo, 0.0)
            new_p = new_probs.get(emo, 0.0)
            self.smoothed_probs[emo] = round(float(self.ema_alpha * new_p + (1.0 - self.ema_alpha) * old_p), 4)

        # Re-normalize to sum to 1.0
        tot = sum(self.smoothed_probs.values())
        if tot > 0:
            self.smoothed_probs = {k: round(v / tot, 4) for k, v in self.smoothed_probs.items()}

        # 2. Hysteresis decision: Should we switch the displayed primary emotion?
        top_emo = max(self.smoothed_probs, key=self.smoothed_probs.get)
        top_val = self.smoothed_probs[top_emo]
        current_val = self.smoothed_probs.get(self.current_stable_emotion, 0.0)

        if top_emo == self.current_stable_emotion:
            self.candidate_emotion = None
            self.candidate_consecutive_count = 0
        else:
            # If the candidate clearly beats the current emotion by the hysteresis margin, switch immediately
            if top_val >= (current_val + self.hysteresis_margin):
                self.current_stable_emotion = top_emo
                self.candidate_emotion = None
                self.candidate_consecutive_count = 0
            else:
                # Otherwise require 3 consecutive frames of candidate dominance
                if self.candidate_emotion == top_emo:
                    self.candidate_consecutive_count += 1
                    if self.candidate_consecutive_count >= 3:
                        self.current_stable_emotion = top_emo
                        self.candidate_emotion = None
                        self.candidate_consecutive_count = 0
                else:
                    self.candidate_emotion = top_emo
                    self.candidate_consecutive_count = 1

        return self.smoothed_probs, self.current_stable_emotion

    # ─────────────────────────────────────────────────────────────────────────
    # Environmental & Quality Guidance
    # ─────────────────────────────────────────────────────────────────────────

    def _assess_environment(self, quality_metrics: Dict, primary_face: Optional[Dict]) -> Tuple[str, str]:
        """
        Assesses environmental conditions (lighting, blur, position) and provides
        gentle, plain-language guidance for elderly users.
        """
        brightness = quality_metrics.get('brightness', 100.0)
        sharpness = quality_metrics.get('sharpness', 50.0)

        if brightness < 30.0:
            return "low_light", "Room lighting is dim. Turning on a nearby light may help."
        if sharpness < 18.0:
            return "blurry", "Camera image appears a bit blurry. Please hold steady."

        if primary_face:
            tilt = primary_face.get('quality', {}).get('tilt_angle', 0.0)
            if tilt > 28.0:
                return "tilted", "Facing the camera upright works best."
            fw = primary_face.get('bbox', (0, 0, 100, 100))[2]
            if fw < 50:
                return "far", "You appear a bit far. Moving slightly closer may help."

        return "optimal", "Lighting and camera angle are good."

    def _default_no_face_result(self, calib_progress: float) -> Dict:
        """Returns clean 'no face' result when user leaves the frame."""
        probs = {e: 0.0 for e in EmotionClassifier.TARGET_CLASSES}
        probs['Neutral'] = 1.0

        return {
            'face_detected': False,
            'primary_emotion': 'Neutral',
            'confidence': 0.0,
            'confidence_level': 'Unclear',
            'is_confident': False,
            'display_label': 'No face detected',
            'status_message': 'No face visible. Please position yourself in front of the camera.',
            'guidance_message': 'Please look directly at the camera to begin.',
            'icon': '👤',
            'probabilities': probs,
            'raw_probabilities': probs,
            'valence': 0.0,
            'arousal': 0.0,
            'emotion_type': 'neutral',
            'bounding_box': None,
            'expanded_bbox': None,
            'enhanced_face': None,
            'quality_metrics': {'brightness': 0.0, 'contrast': 0.0, 'sharpness': 0.0, 'is_usable': False},
            'support_checkin': False,
            'support_message': '',
            'baseline_calibrated': self.classifier.baseline_calibrated,
            'calibration_progress': calib_progress,
            'geo_debug': {'dW': None, 'dH': None, 'is_resting': False, 'geo_active': False}
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Main Processing Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, enable_enhancement: bool = True) -> Dict:
        """
        Runs the full pipeline on a single webcam frame.
        """
        calib_progress = min(
            1.0,
            len(self.classifier.calibration_frames) / max(1, self.classifier.calibration_target)
        )

        if frame is None or frame.size == 0:
            return self._default_no_face_result(calib_progress)

        # ── Step 1: Detect primary face ───────────────────────────────────────
        primary_face = self.detector.get_primary_face(frame)

        if primary_face is None:
            self.missing_face_count += 1
            # If face has been missing for over 10 frames (~0.7s), clear smoothed history
            if self.missing_face_count > 10:
                self.smoothed_probs = None
                self.current_stable_emotion = "Neutral"
                self.consecutive_support_frames = 0
            return self._default_no_face_result(calib_progress)

        # Face detected: reset missing face counter
        self.missing_face_count = 0

        aligned_crop = primary_face.get('aligned_face')
        face_crop = aligned_crop if (aligned_crop is not None and getattr(aligned_crop, 'size', 0) > 0) else primary_face['cropped_face']
        landmarks = primary_face.get('landmarks')

        # ── Step 2: Quality assessment & adaptive enhancement ─────────────────
        if enable_enhancement:
            enhanced_face, quality_metrics = self.enhancer.enhance(face_crop)
        else:
            enhanced_face = face_crop
            quality_metrics = self.enhancer.compute_quality_metrics(face_crop)

        env_status, guidance_msg = self._assess_environment(quality_metrics, primary_face)

        # ── Step 3: Baseline calibration phase (first ~35 frames) ─────────────
        if not self.classifier.baseline_calibrated:
            self.classifier.collect_baseline(enhanced_face, landmarks=landmarks)
            calib_progress = min(
                1.0,
                len(self.classifier.calibration_frames) / max(1, self.classifier.calibration_target)
            )
            calib_probs = {e: 0.02 for e in EmotionClassifier.TARGET_CLASSES}
            calib_probs['Neutral'] = 0.88

            return {
                'face_detected': True,
                'primary_emotion': 'Neutral',
                'confidence': 0.88,
                'confidence_level': 'Moderate',
                'is_confident': True,
                'display_label': 'Calibrating...',
                'status_message': 'Please look comfortably at the camera with a relaxed expression.',
                'guidance_message': guidance_msg,
                'icon': '⏳',
                'probabilities': calib_probs,
                'raw_probabilities': calib_probs,
                'valence': 0.0,
                'arousal': 0.0,
                'emotion_type': 'neutral',
                'bounding_box': primary_face['bbox'],
                'expanded_bbox': primary_face['expanded_bbox'],
                'enhanced_face': enhanced_face,
                'quality_metrics': quality_metrics,
                'support_checkin': False,
                'support_message': '',
                'baseline_calibrated': False,
                'calibration_progress': calib_progress,
                'geo_debug': {'dW': 0.0, 'dH': 0.0, 'is_resting': True, 'geo_active': False}
            }

        # ── Step 4: Model inference with honest confidence evaluation ─────────
        raw_pred = self.classifier.predict(enhanced_face, landmarks=landmarks)

        # ── Step 5: Confidence-aware temporal smoothing & hysteresis ──────────
        smoothed_probs, stable_emo = self._update_smoothed_probabilities(raw_pred['probabilities'])
        stable_conf = smoothed_probs.get(stable_emo, 0.0)

        # Map to calibrated confidence level
        if stable_conf < EmotionClassifier.CONFIDENCE_THRESHOLD_MIN:
            confidence_level = "Unclear"
            display_label = "Expression unclear"
            status_msg = "Unable to identify the expression confidently."
            icon = "🤔"
            is_confident = False
        elif stable_conf < 0.65:
            confidence_level = "Moderate"
            display_label = f"Possible: {stable_emo}"
            status_msg = f"Subtle {stable_emo.lower()} expression observed."
            icon = EmotionClassifier.EMOTION_DYNAMICS.get(stable_emo, {}).get('icon', '😐')
            is_confident = True
        else:
            confidence_level = "High"
            display_label = stable_emo
            status_msg = f"Clear {stable_emo.lower()} expression visible."
            icon = EmotionClassifier.EMOTION_DYNAMICS.get(stable_emo, {}).get('icon', '😐')
            is_confident = True

        # ── Step 6: Calm Wellbeing Check-in (Non-alarmist) ─────────────────────
        # Only triggers if prolonged distress/sadness is sustained for > 150 frames (~10s)
        neg_mass = sum(smoothed_probs.get(e, 0.0) for e in self.SUPPORT_CHECKIN_EMOTIONS)
        if neg_mass >= 0.65:
            self.consecutive_support_frames += 1
        else:
            self.consecutive_support_frames = max(0, self.consecutive_support_frames - 2)

        support_checkin = (self.consecutive_support_frames >= 150)
        support_msg = (
            "We notice you might be feeling down or distressed. Remember you are not alone — would you like to take a break or connect with family?"
            if support_checkin else ""
        )

        return {
            'face_detected': True,
            'primary_emotion': stable_emo,
            'confidence': float(stable_conf),
            'confidence_level': confidence_level,
            'is_confident': is_confident,
            'display_label': display_label,
            'status_message': status_msg,
            'guidance_message': guidance_msg,
            'icon': icon,
            'probabilities': smoothed_probs,
            'raw_probabilities': raw_pred['probabilities'],
            'valence': raw_pred.get('valence', 0.0),
            'arousal': raw_pred.get('arousal', 0.0),
            'emotion_type': raw_pred.get('emotion_type', 'neutral'),
            'bounding_box': primary_face['bbox'],
            'expanded_bbox': primary_face['expanded_bbox'],
            'enhanced_face': enhanced_face,
            'quality_metrics': quality_metrics,
            'support_checkin': support_checkin,
            'support_message': support_msg,
            'baseline_calibrated': True,
            'calibration_progress': 1.0,
            'geo_debug': raw_pred.get('geo_debug', {})
        }
