"""
DOSTH AI — Elderly Model Setup (Auto-Download + Train)
=======================================================
ONE script that does everything:
  1. Downloads FER2013 dataset from HuggingFace (free, no registration)
  2. Applies elderly-domain weighting to training samples
  3. Extracts features using the FER+ ONNX model
  4. Trains an SVM correction layer optimized for elderly faces
  5. Saves the model — ready to use immediately in the web UI

Dataset used:
  • FER2013 (via HuggingFace) — 35,887 labeled face images, 7 emotions
  • Elderly-proxy weighting: samples with lower spatial frequency
    (i.e., smoother, smaller expression deltas — matching elderly face patterns)
    are given 3x weight during SVM training

Run:
    .\\venv\\Scripts\\python scripts\\setup_elderly_model.py

Takes about: 3-8 minutes on CPU depending on internet speed.
"""

import os
import sys
import numpy as np
import cv2

# ─── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.fer.emotion_model import EmotionClassifier
from modules.fer.elderly_adapter import ElderlyAdapter, EMOTIONS_7

# ─── Emotion label maps ───────────────────────────────────────────────────────
# FER2013 label order (standard)
FER2013_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# HuggingFace dataset label order for trpakov/face-expression-recognition
TRPAKOV_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def print_header():
    print("\n" + "=" * 62)
    print("   DOSTH AI — Elderly Emotion Model Setup")
    print("   Downloading + Training (no manual steps needed)")
    print("=" * 62)


def check_dependencies():
    missing = []
    try:
        import datasets
    except ImportError:
        missing.append("datasets")
    try:
        from sklearn.svm import SVC
    except ImportError:
        missing.append("scikit-learn")
    if missing:
        print(f"\n[ERROR] Missing packages: {missing}")
        print("Run: .\\venv\\Scripts\\pip install " + " ".join(missing))
        sys.exit(1)
    print("[OK] All dependencies available.")


def download_fer_dataset():
    """
    Downloads FER emotion dataset from HuggingFace.
    Tries multiple sources in order — uses whichever works first.
    """
    from datasets import load_dataset

    sources = [
        ("trpakov/face-expression-recognition", "full",   TRPAKOV_LABELS),
        ("Preds/fer2013",                        "default", FER2013_LABELS),
        ("asif00/FER_dataset",                   "default", FER2013_LABELS),
    ]

    for repo_id, config, label_map in sources:
        try:
            print(f"\n[INFO] Trying dataset: {repo_id} ...")
            ds = load_dataset(repo_id, config, trust_remote_code=True)
            split = ds.get("train") or ds.get("Training") or list(ds.values())[0]
            print(f"  [OK] Loaded {len(split)} samples from {repo_id}")
            return split, label_map
        except Exception as e:
            print(f"  [SKIP] {repo_id}: {e}")

    print("\n[ERROR] Could not download any FER dataset.")
    print("Check your internet connection and try again.")
    sys.exit(1)


def image_to_bgr(img) -> np.ndarray:
    """Converts a PIL Image or numpy array to BGR numpy array for OpenCV."""
    if hasattr(img, 'convert'):          # PIL Image
        img = img.convert('RGB')
        arr = np.array(img, dtype=np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img
    return None


def compute_elderly_weight(face_bgr: np.ndarray) -> float:
    """
    Computes an 'elderly proxy' weight for a face image.

    Elderly faces have:
    - Smaller spatial frequency in expression regions (subtle movements)
    - Lower global contrast compared to exaggerated training expressions
    - More uniform texture (wrinkles blend into face at 64x64 resolution)

    Faces with LOW expression intensity (small edge response) get higher
    weight — they represent the elderly expression pattern we want to learn.

    Returns weight in [1.0, 3.0]
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if len(face_bgr.shape) == 3 else face_bgr
    resized = cv2.resize(gray, (48, 48)).astype(np.float32)

    # Measure expression intensity via Laplacian edge magnitude
    lap = cv2.Laplacian(resized, cv2.CV_32F)
    edge_magnitude = float(np.mean(np.abs(lap)))

    # Lower edge = more subtle expression = higher elderly weight
    # Edge magnitude: ~5 (very subtle) to ~40 (very exaggerated)
    weight = 1.0 + 2.0 * max(0.0, (20.0 - edge_magnitude) / 20.0)
    return min(3.0, max(1.0, weight))


def extract_features(split, label_map: list, classifier: EmotionClassifier,
                     max_per_class: int = 800):
    """
    Extracts FER+ raw logit features from dataset images.
    Returns X (features), y (labels), weights (per-sample elderly weights).
    """
    from collections import defaultdict

    X, y, weights = [], [], []
    class_counts = defaultdict(int)

    print(f"\n[INFO] Extracting features (max {max_per_class} per class)...")
    print("  This uses FER+ ONNX as a feature extractor.")

    # Find the image and label column names
    sample = split[0]
    img_key = next((k for k in ('image', 'img', 'Image', 'pixels') if k in sample), None)
    lbl_key = next((k for k in ('label', 'labels', 'Label', 'emotion') if k in sample), None)

    if img_key is None or lbl_key is None:
        print(f"  [ERROR] Unexpected dataset columns: {list(sample.keys())}")
        sys.exit(1)

    print(f"  Image column: '{img_key}' | Label column: '{lbl_key}'")

    total = len(split)
    for i, row in enumerate(split):
        if i % 2000 == 0:
            pct = int(100 * i / total)
            done = {k: v for k, v in class_counts.items()}
            print(f"  [{pct:3d}%] Processed {i}/{total} | Classes: {done}")

        label_idx = row[lbl_key]
        if isinstance(label_idx, str):
            if label_idx in label_map:
                emotion = label_idx
            else:
                continue
        else:
            if label_idx < 0 or label_idx >= len(label_map):
                continue
            emotion = label_map[label_idx]

        if emotion not in EMOTIONS_7:
            continue
        if class_counts[emotion] >= max_per_class:
            continue

        face_bgr = image_to_bgr(row[img_key])
        if face_bgr is None:
            continue

        # Extract raw FER+ logits
        tensor = classifier.preprocess_face(face_bgr)
        raw_logits = classifier.session.run(
            [classifier.output_name],
            {classifier.input_name: tensor}
        )[0][0]

        # Elderly proxy weight — subtle expressions get 3x weight
        w = compute_elderly_weight(face_bgr)

        X.append(raw_logits)
        y.append(emotion)
        weights.append(w)
        class_counts[emotion] += 1

    print(f"\n[INFO] Feature extraction complete.")
    print(f"  Total samples: {len(X)}")
    for emotion in EMOTIONS_7:
        print(f"  {emotion:12s}: {class_counts.get(emotion, 0)}")

    return np.array(X), np.array(y), np.array(weights)


def train_svm(X, y, weights):
    """Trains elderly-weighted SVM on FER+ logit features."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report

    print("\n[INFO] Training elderly-weighted SVM...")
    print("  Kernel: RBF | C=8.0 | class_weight=balanced")
    print("  Elderly-proxy weighting: subtle expressions → 3x weight")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(
        kernel='rbf',
        C=8.0,
        gamma='scale',
        probability=True,
        class_weight='balanced',   # Handles class imbalance
        random_state=42
    )

    # Cross-validation for honest accuracy estimate
    print("\n  Running 5-fold cross-validation...")
    try:
        cv_scores = cross_val_score(
            model, X_scaled, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='balanced_accuracy',
            fit_params={'sample_weight': weights}
        )
        print(f"  Cross-val balanced accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    except Exception as e:
        print(f"  Cross-val skipped ({e})")

    # Final fit with elderly weights
    model.fit(X_scaled, y, sample_weight=weights)
    train_acc = model.score(X_scaled, y)
    print(f"  Training accuracy: {train_acc:.1%}")

    # Classification report
    y_pred = model.predict(X_scaled)
    print("\n  Classification Report (per emotion):")
    print(classification_report(y, y_pred, target_names=sorted(set(y)), zero_division=0))

    return model, scaler


def main():
    print_header()
    check_dependencies()

    # Step 1: Download dataset
    print("\n[STEP 1/4] Downloading FER dataset from HuggingFace...")
    split, label_map = download_fer_dataset()

    # Step 2: Load FER+ feature extractor
    print("\n[STEP 2/4] Loading FER+ ONNX model (feature extractor)...")
    classifier = EmotionClassifier()

    # Step 3: Extract features
    print("\n[STEP 3/4] Extracting features with elderly-proxy weighting...")
    X, y, weights = extract_features(split, label_map, classifier, max_per_class=800)

    if len(X) < 50:
        print("\n[ERROR] Too few samples extracted. Check dataset format.")
        sys.exit(1)

    # Step 4: Train + save
    print("\n[STEP 4/4] Training elderly-optimized SVM adapter...")
    model, scaler = train_svm(X, y, weights)

    adapter = ElderlyAdapter()
    adapter.save(
        model=model,
        scaler=scaler,
        n_samples=len(X),
        classes=sorted(set(y)),
        notes="Auto-trained on FER2013 via HuggingFace with elderly-proxy weighting"
    )

    print("\n" + "=" * 62)
    print("  SUCCESS! Elderly model ready.")
    print("=" * 62)
    print(f"  Trained on: {len(X)} samples")
    print(f"  Saved to:   models/fer/elderly_adapter.pkl")
    print()
    print("  Start the demo now:")
    print("  .\\venv\\Scripts\\python scripts\\run_fer_web_ui.py")
    print("=" * 62 + "\n")


if __name__ == '__main__':
    main()
