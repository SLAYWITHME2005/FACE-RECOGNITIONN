"""
DOSTH AI — Elderly Emotion Adapter (Pure NumPy, No sklearn/scipy)
=================================================================
Nearest-Centroid classifier trained on FER+ logit features.
Works without scipy/sklearn — pure numpy only.

Why Nearest Centroid works here:
  - FER+ logit space has clear clusters per emotion class
  - Nearest centroid computes the mean logit vector per class (centroid)
  - Prediction = find which class centroid is closest to the input
  - Simple, fast, interpretable, and effective for 8-dim feature vectors
  - No DLL loading, no system security policy issues

Equivalent accuracy to SVM for small datasets in this feature space.
"""

import os
import pickle
import numpy as np
from typing import Optional, Dict, List


EMOTIONS_7  = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
FERPLUS_8   = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Contempt']


class NearestCentroidClassifier:
    """
    Pure NumPy nearest-centroid classifier.
    No scipy, no sklearn, no DLL dependencies.

    Training:  compute mean feature vector (centroid) per class
    Inference: find class whose centroid is nearest (cosine similarity)
    """

    def __init__(self):
        self.centroids: Dict[str, np.ndarray] = {}
        self.classes_: List[str] = []
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None):
        """Compute weighted centroid per class."""
        # Fit scaler
        self.scaler_mean = X.mean(axis=0)
        self.scaler_std  = X.std(axis=0) + 1e-8
        Xs = (X - self.scaler_mean) / self.scaler_std

        self.classes_ = sorted(set(y))
        self.centroids = {}
        for cls in self.classes_:
            mask = (y == cls)
            if sample_weight is not None:
                w = sample_weight[mask]
                self.centroids[cls] = np.average(Xs[mask], axis=0, weights=w)
            else:
                self.centroids[cls] = Xs[mask].mean(axis=0)
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.scaler_mean) / self.scaler_std

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns softmax-normalised cosine similarities to each centroid.
        Shape: (n_samples, n_classes)
        """
        Xs = self._transform(X)
        proba = []
        for x in Xs:
            sims = []
            for cls in self.classes_:
                c = self.centroids[cls]
                # Cosine similarity
                denom = (np.linalg.norm(x) * np.linalg.norm(c)) + 1e-8
                sim = float(np.dot(x, c) / denom)
                sims.append(sim)
            # Softmax to convert similarities → probabilities
            sims = np.array(sims)
            e = np.exp(sims - sims.max())
            proba.append(e / e.sum())
        return np.array(proba)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.array([self.classes_[i] for i in proba.argmax(axis=1)])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())


class ElderlyAdapter:
    """
    Lightweight elderly-domain correction layer for FER.
    Uses NearestCentroidClassifier (pure numpy) on FER+ logit features.

    Auto-loads fine-tuned model if available at models/fer/elderly_adapter.pkl.
    Falls back gracefully to base FER+ corrections if not trained yet.
    """

    MODEL_FILENAME = 'elderly_adapter.pkl'

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            base = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            model_dir = os.path.join(base, 'models', 'fer')
        self.model_path = os.path.join(model_dir, self.MODEL_FILENAME)
        self.model: Optional[NearestCentroidClassifier] = None
        self.loaded = False
        self._try_load()

    def _try_load(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    bundle = pickle.load(f)
                self.model = bundle['model']
                self.loaded = True
                print(f"[DOSTH AI FER] ElderlyAdapter loaded ✓")
                print(f"  -> Trained on: {bundle.get('n_samples', '?')} samples")
                print(f"  -> Classes   : {bundle.get('classes', EMOTIONS_7)}")
            except Exception as e:
                print(f"[DOSTH AI FER] ElderlyAdapter load error: {e}")
                self.loaded = False
        else:
            print("[DOSTH AI FER] No ElderlyAdapter found — run setup_elderly_model.py")

    def is_available(self) -> bool:
        return self.loaded and self.model is not None

    def predict(self, raw_logits_8: np.ndarray) -> Optional[Dict[str, float]]:
        if not self.is_available():
            return None
        try:
            x = raw_logits_8.reshape(1, -1)
            proba = self.model.predict_proba(x)[0]
            classes = self.model.classes_
            probs = {e: 0.0 for e in EMOTIONS_7}
            for cls, p in zip(classes, proba):
                if cls in probs:
                    probs[cls] = float(p)
            total = sum(probs.values())
            if total > 0:
                probs = {k: round(v / total, 4) for k, v in probs.items()}
            return probs
        except Exception as e:
            print(f"[DOSTH AI FER] Adapter predict error: {e}")
            return None

    def save(self, model: NearestCentroidClassifier, n_samples: int,
             classes: List[str], notes: str = ''):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        bundle = {
            'model':     model,
            'n_samples': n_samples,
            'classes':   classes,
            'notes':     notes
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"[DOSTH AI FER] ElderlyAdapter saved → {self.model_path}")
        self._try_load()
