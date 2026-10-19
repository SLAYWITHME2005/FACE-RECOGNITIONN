"""
DOSTH AI — Face Detection Module (face_detector.py)
=====================================================
YuNet-based deep learning face detector, fully upgraded for the
Malayali Elderly Care domain.

Key Features:
    1. YuNet ONNX model — fastest deep-learning face detector for CPU
    2. Multi-scale detection — detects faces at varying distances from camera
    3. Face alignment via eye landmarks — corrects tilted/rotated faces
       before passing to the emotion classifier
    4. Quality scoring — checks blur, face size, and pose to reject bad crops
    5. Face stability tracking — smooths bounding box across frames to
       prevent jitter (important for elderly who move slowly)
    6. Elderly-tuned defaults — lower score threshold (0.50), larger margin
       (22%) to capture full facial context including forehead wrinkles

YuNet output format per detected face (15 values):
    [x, y, w, h,
     right_eye_x, right_eye_y,
     left_eye_x,  left_eye_y,
     nose_x,      nose_y,
     right_mouth_x, right_mouth_y,
     left_mouth_x,  left_mouth_y,
     confidence_score]
"""

import os
import math
import urllib.request
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


class FaceDetector:
    """
    Robust deep-learning face detector for DOSTH AI.

    Uses YuNet (ONNX) via OpenCV FaceDetectorYN with:
      - Eye-landmark based alignment
      - Quality-based filtering (blur, size, pose angle)
      - Bounding box smoothing across frames (EMA tracker)
      - Multi-scale fallback for small/distant faces
    """

    YUNET_URL = (
        'https://github.com/opencv/opencv_zoo/raw/main/models/'
        'face_detection_yunet/face_detection_yunet_2023mar.onnx'
    )

    # Minimum face size (pixels) to accept — rejects tiny far-away faces
    MIN_FACE_PX = 40

    # Maximum eye tilt to accept without correction (degrees)
    MAX_TILT_DEG = 30

    def __init__(
        self,
        model_path: Optional[str] = None,
        score_threshold: float = 0.50,   # Lower than default → catches subtle elderly expressions
        nms_threshold: float = 0.30,
        margin_ratio: float = 0.06,      # Tight crop → face fills 64x64 input as FER+ expects
        blur_threshold: float = 35.0,    # Laplacian variance below this = too blurry to use
        smooth_alpha: float = 0.4        # EMA smoothing factor for bounding box stability
    ):
        """
        Args:
            score_threshold: Minimum YuNet confidence to accept a detection.
                             0.50 is tuned for elderly faces (wrinkles reduce model confidence).
            margin_ratio: Fractional expansion around detected face bounding box.
                          0.06 keeps a tight crop so the face fills the 64x64 input
                          as FER+ training data expected. Larger values add hair/neck
                          which biases the CNN toward Happy.
            blur_threshold: Laplacian variance threshold. Frames below this are marked low-quality.
            smooth_alpha: EMA factor for bounding box smoothing. 0.4 = moderate smoothing.
        """
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.margin_ratio = margin_ratio
        self.blur_threshold = blur_threshold
        self.smooth_alpha = smooth_alpha

        # Resolve model path
        if model_path is None:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            model_dir = os.path.join(base_dir, 'models', 'fer')
            os.makedirs(model_dir, exist_ok=True)
            self.model_path = os.path.join(model_dir, 'face_detection_yunet.onnx')
        else:
            self.model_path = model_path

        self._ensure_model_exists()

        # Initialize YuNet detector
        self.detector = cv2.FaceDetectorYN_create(
            self.model_path,
            "",
            (320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000
        )
        self._current_input_size = (320, 320)

        # EMA smoothed bounding box for primary face (reduces jitter)
        self._smoothed_bbox: Optional[np.ndarray] = None

        print("[DOSTH AI FaceDetector] YuNet initialized.")
        print(f"  -> Score threshold : {score_threshold}")
        print(f"  -> Face margin     : {int(margin_ratio * 100)}%")
        print(f"  -> Blur threshold  : {blur_threshold}")

    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_model_exists(self):
        """Downloads YuNet ONNX model if not present or corrupted."""
        if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 100_000:
            print(f"[DOSTH AI] Downloading YuNet model → {self.model_path} ...")
            req = urllib.request.Request(
                self.YUNET_URL, headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as resp, open(self.model_path, 'wb') as f:
                f.write(resp.read())
            print("[DOSTH AI] YuNet download complete.")

    def _set_input_size(self, w: int, h: int):
        """Updates detector input size only when frame dimensions change."""
        if self._current_input_size != (w, h):
            self.detector.setInputSize((w, h))
            self._current_input_size = (w, h)

    # ─────────────────────────────────────────────────────────────────────────
    # Quality checks
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_blur_score(self, gray_crop: np.ndarray) -> float:
        """
        Measures image sharpness using Laplacian variance.
        Higher = sharper. Below blur_threshold = too blurry for emotion inference.
        """
        return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

    def _compute_eye_angle(self, landmarks: Dict) -> float:
        """
        Computes the tilt angle (degrees) between left and right eye centers.
        0 degrees = perfectly level face.
        """
        rx, ry = landmarks['right_eye']
        lx, ly = landmarks['left_eye']
        dx = lx - rx
        dy = ly - ry
        return math.degrees(math.atan2(dy, dx))

    def _compute_quality(self, crop: np.ndarray, fw: int, fh: int,
                         landmarks: Dict) -> Dict:
        """
        Computes quality metrics for a detected face crop.

        Returns:
            blur_score    : float — Laplacian variance (higher = sharper)
            tilt_angle    : float — eye tilt in degrees
            face_size_ok  : bool  — face is large enough to classify
            blur_ok       : bool  — face is sharp enough
            pose_ok       : bool  — face is not too tilted
            is_usable     : bool  — all three quality gates passed
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        blur = self._compute_blur_score(gray)
        tilt = abs(self._compute_eye_angle(landmarks))

        size_ok = fw >= self.MIN_FACE_PX and fh >= self.MIN_FACE_PX
        blur_ok = blur >= self.blur_threshold
        pose_ok = tilt <= self.MAX_TILT_DEG

        return {
            'blur_score': round(blur, 1),
            'tilt_angle': round(tilt, 1),
            'face_size_ok': size_ok,
            'blur_ok': blur_ok,
            'pose_ok': pose_ok,
            'is_usable': size_ok and blur_ok and pose_ok
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Alignment
    # ─────────────────────────────────────────────────────────────────────────

    def align_face(self, face_crop: np.ndarray, landmarks: Dict,
                   output_size: int = 112, offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Aligns a face crop by rotating it so the eyes are horizontal.
        Uses local coordinates relative to the crop origin (offset = (x1, y1)).

        Args:
            face_crop   : Cropped face image (BGR)
            landmarks   : Landmark dict (right_eye, left_eye, etc.)
            output_size : Output square size in pixels (default 112x112)
            offset      : (x1, y1) top-left origin of face_crop in full frame
        """
        ox, oy = offset
        rx = landmarks['right_eye'][0] - ox
        ry = landmarks['right_eye'][1] - oy
        lx = landmarks['left_eye'][0] - ox
        ly = landmarks['left_eye'][1] - oy

        # Eye center in local crop coordinates
        eye_center = ((rx + lx) / 2.0, (ry + ly) / 2.0)

        # Compute tilt angle
        angle = math.degrees(math.atan2(ly - ry, lx - rx))

        # If head is already nearly level (< 2.5 deg) or excessive (> 45 deg), keep clean crop without warping
        if abs(angle) < 2.5 or abs(angle) > 45.0:
            return cv2.resize(face_crop, (output_size, output_size), interpolation=cv2.INTER_AREA)

        # Build local rotation matrix and apply within crop boundary
        h, w = face_crop.shape[:2]
        M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        aligned = cv2.warpAffine(
            face_crop, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return cv2.resize(aligned, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # ─────────────────────────────────────────────────────────────────────────
    # Bounding box smoothing
    # ─────────────────────────────────────────────────────────────────────────

    def _smooth_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Applies Exponential Moving Average to bounding box coordinates
        to prevent the face box from jittering frame-to-frame.
        Especially useful for elderly users who move slowly and steadily.
        """
        new = np.array(bbox, dtype=np.float32)
        if self._smoothed_bbox is None:
            self._smoothed_bbox = new
        else:
            self._smoothed_bbox = (
                self.smooth_alpha * new + (1.0 - self.smooth_alpha) * self._smoothed_bbox
            )
        return tuple(self._smoothed_bbox.astype(int))

    def reset_tracker(self):
        """Resets the bounding box smoother (call on new user or scene change)."""
        self._smoothed_bbox = None

    # ─────────────────────────────────────────────────────────────────────────
    # Main detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detects all faces in a frame.

        Processing steps:
          1. Run YuNet on full frame
          2. If no faces found → retry at 0.5x scale (multi-scale fallback)
          3. For each detection: expand margin, crop, compute quality metrics
          4. Sort by face area (largest = closest = primary user)

        Returns list of face dicts, each containing:
            bbox          : (x, y, w, h) original tight bounding box
            expanded_bbox : (x1, y1, x2, y2) expanded with margin
            smoothed_bbox : (x, y, w, h) EMA-smoothed bounding box
            confidence    : float [0–1] YuNet detection confidence
            area          : int — pixel area of original bbox
            cropped_face  : np.ndarray — expanded face crop (BGR)
            aligned_face  : np.ndarray — eye-aligned 112x112 crop (BGR)
            landmarks     : dict with right_eye, left_eye, nose, right_mouth, left_mouth
            quality       : dict with blur_score, tilt_angle, is_usable etc.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        self._set_input_size(w, h)

        _, raw_faces = self.detector.detect(frame)

        # ── Multi-scale fallback: retry at half resolution if nothing found ──
        if raw_faces is None or len(raw_faces) == 0:
            small = cv2.resize(frame, (w // 2, h // 2))
            sh, sw = small.shape[:2]
            self._set_input_size(sw, sh)
            _, raw_faces = self.detector.detect(small)
            self._set_input_size(w, h)  # Restore full size

            if raw_faces is not None and len(raw_faces) > 0:
                # Scale detections back to full frame coordinates
                scale_x, scale_y = w / sw, h / sh
                scaled = []
                for face in raw_faces:
                    f = face.copy()
                    # Scale x,y,w,h and all landmark coords
                    for i in [0, 2, 4, 6, 8, 10, 12]:
                        f[i] *= scale_x
                    for i in [1, 3, 5, 7, 9, 11, 13]:
                        f[i] *= scale_y
                    scaled.append(f)
                raw_faces = np.array(scaled)
            else:
                return []

        results = []
        for face in raw_faces:
            x   = int(face[0]);   y  = int(face[1])
            fw  = int(face[2]);   fh = int(face[3])
            conf = float(face[14]) if len(face) > 14 else float(face[-1])

            # Skip tiny detections
            if fw < self.MIN_FACE_PX or fh < self.MIN_FACE_PX:
                continue

            # ── Expand bounding box with margin ──────────────────────────────
            mx = int(fw * self.margin_ratio)
            my = int(fh * self.margin_ratio)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + fw + mx)
            y2 = min(h, y + fh + my)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ── Extract landmarks ─────────────────────────────────────────────
            landmarks = {
                'right_eye':   (int(face[4]),  int(face[5])),
                'left_eye':    (int(face[6]),  int(face[7])),
                'nose':        (int(face[8]),  int(face[9])),
                'right_mouth': (int(face[10]), int(face[11])),
                'left_mouth':  (int(face[12]), int(face[13]))
            }

            # ── Compute quality metrics ───────────────────────────────────────
            quality = self._compute_quality(crop, fw, fh, landmarks)

            # ── Eye-aligned face crop ─────────────────────────────────────────
            try:
                aligned = self.align_face(crop, landmarks, output_size=112, offset=(x1, y1))
            except Exception:
                aligned = cv2.resize(crop, (112, 112))

            # ── Smooth bounding box ───────────────────────────────────────────
            smoothed = self._smooth_bbox((x, y, fw, fh))

            results.append({
                'bbox':          (x, y, fw, fh),
                'expanded_bbox': (x1, y1, x2, y2),
                'smoothed_bbox': smoothed,
                'confidence':    round(conf, 3),
                'area':          fw * fh,
                'cropped_face':  crop,
                'aligned_face':  aligned,
                'landmarks':     landmarks,
                'quality':       quality
            })

        # Sort largest (closest) face first — that's the primary user
        results.sort(key=lambda f: f['area'], reverse=True)
        return results

    def get_primary_face(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Returns only the primary (largest, closest) detected face.
        Returns None if no face is found.
        """
        faces = self.detect_faces(frame)
        return faces[0] if faces else None

    def get_usable_face(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Returns the primary face only if it passes all quality checks.
        Rejects blurry, tiny, or extremely tilted faces.
        Returns None if no usable face found.
        """
        face = self.get_primary_face(frame)
        if face and face['quality']['is_usable']:
            return face
        return None
