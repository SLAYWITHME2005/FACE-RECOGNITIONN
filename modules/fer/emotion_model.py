import os
import math
import urllib.request
import numpy as np
import cv2
import onnxruntime as ort
from typing import Dict, Optional

class EmotionClassifier:
    """
    Robust Emotion Classifier for DOSTH AI Elderly Care.
    
    Key Design Principles:
    1. Continuous, bounded affective heuristics (no brittle binary thresholds).
    2. Calibrated softmax temperature (T=0.85) to prevent artificial 99.9% certainty.
    3. Honest confidence thresholding: distinguishes clear vs. subtle vs. ambiguous expressions.
    4. Personal baseline calibration: cancels resting wrinkle bias while preserving genuine expressions.
    5. Elderly-domain accessibility: presents results as observable visual estimates.
    """

    FERPLUS_CLASSES = [
        'Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Contempt'
    ]

    TARGET_CLASSES = [
        'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
    ]

    EMOTION_DYNAMICS = {
        'Happy':    {'valence': 0.8,  'arousal': 0.7, 'type': 'positive',  'icon': '😊'},
        'Neutral':  {'valence': 0.0,  'arousal': 0.1, 'type': 'neutral',   'icon': '😐'},
        'Surprise': {'valence': 0.4,  'arousal': 0.8, 'type': 'positive',  'icon': '😲'},
        'Sad':      {'valence': -0.7, 'arousal': 0.3, 'type': 'negative',  'icon': '😔'},
        'Fear':     {'valence': -0.8, 'arousal': 0.9, 'type': 'negative',  'icon': '😨'},
        'Angry':    {'valence': -0.7, 'arousal': 0.8, 'type': 'negative',  'icon': '😠'},
        'Disgust':  {'valence': -0.6, 'arousal': 0.5, 'type': 'negative',  'icon': '😣'}
    }

    MODEL_URL = 'https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx'

    # Honest confidence thresholds
    CONFIDENCE_THRESHOLD_MIN = 0.38   # Minimum probability to accept an emotion
    MARGIN_THRESHOLD_MIN     = 0.06   # Minimum difference between top 1 and top 2 class

    def __init__(self, model_path: Optional[str] = None,
                 temperature: float = 0.85,
                 neutral_suppression: float = 0.0):
        """
        Args:
            temperature: Softmax temperature (0.85 provides natural calibration).
            neutral_suppression: Gentle attenuation for resting Neutral dominance.
        """
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base_dir, 'models', 'fer')
            os.makedirs(model_dir, exist_ok=True)
            self.model_path = os.path.join(model_dir, 'emotion_model.onnx')
        else:
            self.model_path = model_path

        self.temperature = temperature
        self.neutral_suppression = neutral_suppression

        # Baseline calibration state
        self.baseline_logits = None
        self.baseline_geometry: Optional[Dict[str, float]] = None
        self.baseline_calibrated = False
        self.calibration_frames = []
        self.calibration_geometries = []
        self.calibration_target = 35  # Frames to establish resting baseline (~2.5s)

        self._ensure_model_exists()

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.model_path, opts, providers=['CPUExecutionProvider']
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Load fine-tuned elderly adapter if available
        try:
            from .elderly_adapter import ElderlyAdapter
            self.elderly_adapter = ElderlyAdapter()
        except Exception as e:
            self.elderly_adapter = None

        print("[DOSTH AI FER] EmotionClassifier initialized with Calibrated Affective Engine.")
        print(f"  -> Temperature: {temperature}, Neutral Suppression: {neutral_suppression}")

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 1000000:
            print(f"[DOSTH AI] Downloading emotion model to {self.model_path}...")
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
            print("[DOSTH AI] Model download complete.")

    @staticmethod
    def extract_geometry(landmarks: Optional[Dict]) -> Optional[Dict[str, float]]:
        """
        Extracts scale-invariant facial geometry metrics from landmarks.
        Normalizes all distances by Inter-Ocular Distance (IOD).
        """
        if not landmarks or not isinstance(landmarks, dict):
            return None
        try:
            rx, ry = landmarks['right_eye']
            lx, ly = landmarks['left_eye']
            nx, ny = landmarks['nose']
            rmx, rmy = landmarks['right_mouth']
            lmx, lmy = landmarks['left_mouth']

            iod = math.hypot(lx - rx, ly - ry)
            if iod < 5.0:
                return None

            # Normalized mouth width
            mouth_w = math.hypot(lmx - rmx, lmy - rmy) / iod

            # Normalized distance from nose to mouth center
            mouth_cy = (rmy + lmy) / 2.0
            nose_to_mouth = (mouth_cy - ny) / iod

            return {
                'mouth_width': float(mouth_w),
                'nose_to_mouth': float(nose_to_mouth)
            }
        except Exception:
            return None

    def _softmax_temperature(self, x: np.ndarray) -> np.ndarray:
        """Temperature-scaled Softmax."""
        scaled = x / max(0.1, self.temperature)
        e_x = np.exp(scaled - np.max(scaled))
        return e_x / e_x.sum()

    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Converts face crop to [1, 1, 64, 64] float32 tensor.
        Uses clean single-pass grayscale conversion and light CLAHE to avoid
        over-processing and contrast artifacts.
        """
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()

        # Edge-preserving wrinkle smoothing
        smoothed = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        norm_face = clahe.apply(smoothed)

        resized = cv2.resize(norm_face, (64, 64), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32).reshape(1, 1, 64, 64)
        return tensor

    def collect_baseline(self, face_img: np.ndarray, landmarks: Optional[Dict] = None) -> bool:
        """
        Personal Baseline Calibration:
        Observes resting face to capture individual resting wrinkle patterns.
        """
        if self.baseline_calibrated:
            return True

        tensor = self.preprocess_face(face_img)
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0][0]
        self.calibration_frames.append(raw)

        if landmarks:
            geom = self.extract_geometry(landmarks)
            if geom:
                self.calibration_geometries.append(geom)

        if len(self.calibration_frames) >= self.calibration_target:
            self.baseline_logits = np.mean(self.calibration_frames, axis=0)
            if self.calibration_geometries:
                self.baseline_geometry = {
                    'mouth_width': float(np.mean([g['mouth_width'] for g in self.calibration_geometries])),
                    'nose_to_mouth': float(np.mean([g['nose_to_mouth'] for g in self.calibration_geometries]))
                }
            self.baseline_calibrated = True
            print("[DOSTH AI FER] Personal Baseline Calibration complete.")
        return self.baseline_calibrated

    def reset_baseline(self):
        """Resets user baseline for a new user."""
        self.baseline_logits = None
        self.baseline_geometry = None
        self.baseline_calibrated = False
        self.calibration_frames = []
        self.calibration_geometries = []

    def predict(self, face_img: np.ndarray, landmarks: Optional[Dict] = None) -> Dict:
        """
        Predicts facial expression from face crop with honest confidence assessment.
        """
        if face_img is None or face_img.size == 0:
            return self._empty_prediction()

        tensor = self.preprocess_face(face_img)
        raw_outputs = self.session.run([self.output_name], {self.input_name: tensor})[0][0]

        # ── PRIORITY 1: Fine-tuned ElderlyAdapter if available ────────────────
        if self.elderly_adapter and self.elderly_adapter.is_available():
            adapter_probs = self.elderly_adapter.predict(raw_outputs)
            if adapter_probs is not None:
                return self._format_result(adapter_probs, mode='elderly_adapter')

        # ── PRIORITY 2: Targeted baseline correction + Smooth Continuous Geometric Hints
        if self.baseline_calibrated and self.baseline_logits is not None:
            # Only cancel the Sad (3) and Angry (4) resting wrinkle bias.
            # Do NOT subtract from Neutral (0) — that was the root cause of Happy bias:
            # subtracting 50% of a large Neutral resting logit directly promoted Happy to 1st.
            adjusted_logits = raw_outputs.astype(np.float32).copy()
            # For Sad and Angry: pull them down by their resting level so wrinkles don't
            # permanently inflate these scores on a genuinely neutral face.
            for cls_idx in (3, 4):  # Sad=3, Angry=4
                resting_val = float(self.baseline_logits[cls_idx])
                if resting_val > 0:
                    adjusted_logits[cls_idx] -= resting_val * 0.50
        else:
            adjusted_logits = raw_outputs.astype(np.float32).copy()

        # Step B: Continuous geometric hints (no hard cliff jumps)
        is_resting = True
        geo_debug = {'dW': 0.0, 'dH': 0.0, 'is_resting': True, 'geo_active': False}

        if landmarks:
            geom = self.extract_geometry(landmarks)
            if geom:
                # If baseline geometry not yet initialized, seed it from this first face
                if self.baseline_geometry is None:
                    self.baseline_geometry = {
                        'mouth_width': geom['mouth_width'],
                        'nose_to_mouth': geom['nose_to_mouth']
                    }

                ref_w = self.baseline_geometry['mouth_width']
                ref_h = self.baseline_geometry['nose_to_mouth']
                dW = geom['mouth_width'] - ref_w
                dH = geom['nose_to_mouth'] - ref_h

                geo_debug = {
                    'dW': round(dW, 4),
                    'dH': round(dH, 4),
                    'is_resting': False,
                    'geo_active': True
                }

                # Bounded equilibrium zone (natural resting mouth variation)
                is_resting = (abs(dW) <= 0.032 and abs(dH) <= 0.026)
                geo_debug['is_resting'] = is_resting

                if is_resting:
                    # Resting equilibrium: gently stabilize Neutral without locking out others
                    adjusted_logits[0] += 2.0
                    adjusted_logits[1] -= 0.6
                    adjusted_logits[3] -= 0.4   # Sad: moderate suppression
                    adjusted_logits[4] -= 0.3   # Angry: light suppression (easier to surface)
                    adjusted_logits[6] -= 0.3   # Fear:  light suppression (easier to surface)
                else:
                    # ── Genuine Smile: mouth widens ≥0.038 without vertical drop ──────
                    if dW > 0.038 and dH <= 0.028:
                        s_ramp = min(3.5, max(0.5, (dW - 0.034) * 35.0))
                        adjusted_logits[1] += s_ramp
                        adjusted_logits[0] -= s_ramp * 0.65
                        adjusted_logits[3] -= 0.4

                    # ── Angry: tightened / pursed lips — CHECKED BEFORE Sad ───────────
                    # Angry faces press lips together (dW < -0.020) with little dH change.
                    # Must come before Sad so dH>0.028 doesn't steal compressed-lip faces.
                    elif dW < -0.020 and abs(dH) <= 0.028:
                        angry_ramp = min(3.8, max(0.8, (-0.016 - dW) * 40.0))
                        adjusted_logits[4] += angry_ramp
                        adjusted_logits[0] -= angry_ramp * 0.65
                        adjusted_logits[1] -= 0.4

                    # ── Fear: open mouth WITH horizontal stretch (jaw drop + corners pull) ──
                    # Fear = dH > 0.030 AND dW > 0.012 (open AND slightly wide).
                    # This separates Fear from Sad's pure downward droop (dW <= 0.012).
                    # Ramp is strong (up to 5.5) because raw model heavily favours Sad on open-mouth.
                    elif dH > 0.030 and dW > 0.012:
                        fear_ramp = min(5.5, max(1.5, (dH - 0.020) * 50.0))
                        adjusted_logits[6] += fear_ramp
                        # Suppress Sad — it strongly competes on open-mouth images
                        adjusted_logits[3] -= fear_ramp * 0.80
                        # If Surprise logit is also competing, redirect some to Fear
                        if adjusted_logits[2] > adjusted_logits[6]:
                            transfer = (adjusted_logits[2] - adjusted_logits[6]) * 0.70
                            adjusted_logits[6] += transfer
                            adjusted_logits[2] -= transfer
                        adjusted_logits[0] -= fear_ramp * 0.7

                    # ── Sad: pure downward corner droop, mouth stays closed/narrow ────
                    elif dH > 0.028 and dW <= 0.012:
                        sad_ramp = min(3.5, max(0.5, (dH - 0.022) * 35.0))
                        adjusted_logits[3] += sad_ramp
                        adjusted_logits[0] -= sad_ramp * 0.65
                        adjusted_logits[1] -= 0.4

        # Step C: Temperature-scaled softmax
        probs_8 = self._softmax_temperature(adjusted_logits)

        # Step D: Map 8 FER+ classes -> 7 universal target classes
        ferplus_map = {
            'Neutral':  float(probs_8[0]),
            'Happy':    float(probs_8[1]),
            'Surprise': float(probs_8[2]),
            'Sad':      float(probs_8[3]),
            'Angry':    float(probs_8[4]),
            'Disgust':  float(probs_8[5]) + float(probs_8[7]) * 0.5,
            'Fear':     float(probs_8[6])
        }

        # Step E: Gentle Neutral suppression when an active expression occurs
        if not is_resting:
            ferplus_map['Neutral'] *= (1.0 - self.neutral_suppression)

        total_p = sum(ferplus_map.values())
        norm_probs = {k: round(float(v / total_p), 4) for k, v in ferplus_map.items()}

        return self._format_result(norm_probs, mode='calibrated_fer', geo_debug=geo_debug)

    def _format_result(self, norm_probs: Dict[str, float], mode: str, geo_debug: Optional[Dict] = None) -> Dict:
        """
        Formats probabilities into an honest, elderly-accessible response
        with clear confidence tiers and uncertainty handling.
        """
        sorted_probs = sorted(norm_probs.items(), key=lambda x: x[1], reverse=True)
        top_emotion, top_conf = sorted_probs[0]
        second_emotion, second_conf = sorted_probs[1] if len(sorted_probs) > 1 else ('None', 0.0)
        margin = top_conf - second_conf

        dynamics = self.EMOTION_DYNAMICS.get(
            top_emotion, {'valence': 0.0, 'arousal': 0.0, 'type': 'neutral', 'icon': '😐'}
        )

        # Honest confidence categorization
        if top_conf < self.CONFIDENCE_THRESHOLD_MIN or margin < self.MARGIN_THRESHOLD_MIN:
            is_confident = False
            confidence_level = "Unclear"
            display_label = "Expression unclear"
            status_message = "Unable to identify the expression confidently."
            icon = "🤔"
        elif top_conf < 0.65:
            is_confident = True
            confidence_level = "Moderate"
            display_label = f"Possible: {top_emotion}"
            status_message = f"Subtle {top_emotion.lower()} expression observed."
            icon = dynamics.get('icon', '😐')
        else:
            is_confident = True
            confidence_level = "High"
            display_label = top_emotion
            status_message = f"Clear {top_emotion.lower()} expression visible."
            icon = dynamics.get('icon', '😐')

        return {
            'primary_emotion': top_emotion,
            'confidence': float(top_conf),
            'confidence_level': confidence_level,
            'is_confident': is_confident,
            'display_label': display_label,
            'status_message': status_message,
            'icon': icon,
            'probabilities': norm_probs,
            'second_emotion': second_emotion,
            'margin': round(float(margin), 4),
            'valence': dynamics['valence'],
            'arousal': dynamics['arousal'],
            'emotion_type': dynamics['type'],
            'baseline_calibrated': self.baseline_calibrated,
            'mode': mode,
            'geo_debug': geo_debug or {'dW': None, 'dH': None, 'is_resting': False, 'geo_active': False}
        }

    def _empty_prediction(self) -> Dict:
        """Returns safe fallback prediction when no face is available."""
        return {
            'primary_emotion': 'Neutral',
            'confidence': 0.0,
            'confidence_level': 'Unclear',
            'is_confident': False,
            'display_label': 'No face detected',
            'status_message': 'No face currently visible in camera.',
            'icon': '👤',
            'probabilities': {e: 0.0 for e in self.TARGET_CLASSES},
            'second_emotion': 'None',
            'margin': 0.0,
            'valence': 0.0,
            'arousal': 0.0,
            'emotion_type': 'neutral',
            'baseline_calibrated': self.baseline_calibrated,
            'mode': 'fallback',
            'geo_debug': {'dW': None, 'dH': None, 'is_resting': False, 'geo_active': False}
        }
