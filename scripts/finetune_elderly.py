"""
DOSTH AI — Elderly FER Fine-Tuning Script
==========================================
Trains a correction adapter on top of FER+ using collected elderly face images.
No GPU needed — runs entirely on CPU in under 5 minutes.

Pipeline:
    Collected elderly images (JPG)
      → FER+ ONNX model (frozen, feature extractor)
      → 8-dim raw logit features per image
      → StandardScaler normalization
      → SVM with RBF kernel (classifier)
      → Saved as models/fer/elderly_adapter.pkl

Usage:
    python scripts/finetune_elderly.py

    Options:
      --data-dir PATH    Custom path to collected images folder
      --min-samples N    Minimum samples per class (default: 15)

Recommended Data Sources (supplement webcam data):
    1. FACES Dataset (Max Planck Institute) — Ages 58-86, 6 emotions
       URL: https://faces.mpib-berlin.mpg.de/ger/start.html
       → Place images in: data/collected/elderly_faces/<emotion>/
    
    2. RAF-DB (Real-world Affective Faces) — Webcam quality
       URL: http://whdeng.cn/RAF/model1.html
       → After download, run: python scripts/import_rafdb.py
"""

import os
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fer.emotion_model import EmotionClassifier
from modules.fer.elderly_adapter import ElderlyAdapter, EMOTIONS_7

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    import sklearn
    print(f"[INFO] scikit-learn {sklearn.__version__} ready.")
except ImportError:
    print("[ERROR] scikit-learn not installed. Run:")
    print("    .\\venv\\Scripts\\pip install scikit-learn")
    sys.exit(1)


EMOTION_CLASSES = EMOTIONS_7  # 7 target emotions


def load_images_as_features(data_dir: str, classifier: EmotionClassifier,
                              min_samples: int = 15) -> tuple:
    """
    Loads all labeled images, extracts FER+ logit features, returns (X, y).
    
    FER+ logits are used as features (not the softmax output) because they
    contain richer information that the SVM can exploit for elderly correction.
    """
    X = []
    y = []
    class_counts = {}

    print(f"\n[INFO] Loading images from: {data_dir}")
    print("-" * 50)

    for emotion in EMOTION_CLASSES:
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_dir):
            print(f"  [SKIP] {emotion:12s} — folder not found")
            continue

        image_files = [f for f in os.listdir(emotion_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(image_files) < min_samples:
            print(f"  [SKIP] {emotion:12s} — only {len(image_files)} images (need {min_samples}+)")
            continue

        loaded = 0
        for img_file in image_files:
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Extract raw logits from FER+ (before softmax) as feature vector
            tensor = classifier.preprocess_face(img)
            raw_logits = classifier.session.run(
                [classifier.output_name],
                {classifier.input_name: tensor}
            )[0][0]  # shape: (8,)

            X.append(raw_logits)
            y.append(emotion)
            loaded += 1

        class_counts[emotion] = loaded
        print(f"  [OK]   {emotion:12s} — {loaded} images → {loaded} feature vectors")

    print("-" * 50)
    print(f"  Total: {len(X)} samples across {len(class_counts)} classes")

    return np.array(X), np.array(y), class_counts


def train_elderly_adapter(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Trains SVM with RBF kernel on FER+ logit features.
    
    Why SVM?
    - Performs excellently on small datasets (30-500 samples)
    - RBF kernel handles non-linear separation between elderly emotion classes
    - Cross-validation gives reliable accuracy estimates even with small N
    - Much better than random forest / logistic regression for this feature space
    """
    print("\n[INFO] Training SVM classifier...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(
        kernel='rbf',
        C=10.0,           # Regularization — 10.0 works well for 8-dim features
        gamma='scale',    # Auto-scale gamma to 1/(n_features * X.var())
        probability=True, # Enable predict_proba() for probability outputs
        class_weight='balanced',  # Handle class imbalance (e.g., more Neutral samples)
        random_state=42
    )

    # Cross-validation to get honest accuracy estimate
    n_splits = min(5, min(np.bincount(np.searchsorted(np.unique(y), y))))
    n_splits = max(2, n_splits)

    print(f"  → Running {n_splits}-fold cross-validation...")
    try:
        cv_scores = cross_val_score(
            model, X_scaled, y,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
            scoring='balanced_accuracy'
        )
        print(f"  → Cross-val balanced accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    except Exception as e:
        print(f"  → Cross-val skipped: {e}")

    # Train final model on all data
    model.fit(X_scaled, y)
    train_acc = model.score(X_scaled, y)
    print(f"  → Training accuracy: {train_acc:.1%}")

    return model, scaler


def print_confusion_matrix(model, scaler, X, y):
    """Prints a readable confusion matrix for quality check."""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    classes = sorted(set(y))

    print("\n[INFO] Classification Report:")
    print(classification_report(y, y_pred, target_names=classes, zero_division=0))

    print("[INFO] Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y, y_pred, labels=classes)
    header = f"{'':12s}" + "".join(f"{c[:7]:8s}" for c in classes)
    print(header)
    for i, row_class in enumerate(classes):
        row = f"{row_class[:12]:12s}" + "".join(f"{cm[i, j]:8d}" for j in range(len(classes)))
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune elderly emotion adapter')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to collected images folder')
    parser.add_argument('--min-samples', type=int, default=15,
                        help='Minimum samples per emotion class')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(base_dir, 'data', 'collected', 'elderly_faces')

    if not os.path.isdir(data_dir):
        print(f"\n[ERROR] Data directory not found: {data_dir}")
        print("\nFirst collect data using:")
        print("    .\\venv\\Scripts\\python scripts\\collect_training_data.py")
        print("\nOr download the FACES Dataset (ages 58-86) from:")
        print("    https://faces.mpib-berlin.mpg.de/ger/start.html")
        print("And organize it as:")
        print(f"    {data_dir}/Happy/image1.jpg")
        print(f"    {data_dir}/Sad/image1.jpg  ... etc")
        sys.exit(1)

    print("=" * 60)
    print("  DOSTH AI — Elderly FER Adapter Training")
    print("=" * 60)

    # Load base FER+ model for feature extraction
    print("\n[INFO] Loading FER+ model for feature extraction...")
    classifier = EmotionClassifier()

    # Load and extract features from elderly images
    X, y, class_counts = load_images_as_features(data_dir, classifier, args.min_samples)

    if len(X) == 0:
        print("\n[ERROR] No images loaded. Check data directory and minimum sample count.")
        print(f"  Data directory: {data_dir}")
        print(f"  Minimum samples needed per emotion: {args.min_samples}")
        sys.exit(1)

    if len(set(y)) < 2:
        print(f"\n[ERROR] Need at least 2 emotion classes. Found: {set(y)}")
        sys.exit(1)

    # Train the adapter
    model, scaler = train_elderly_adapter(X, y)

    # Quality check
    print_confusion_matrix(model, scaler, X, y)

    # Save adapter
    adapter = ElderlyAdapter()
    adapter.save(
        model=model,
        scaler=scaler,
        n_samples=len(X),
        classes=sorted(set(y)),
        notes=f"Trained on {len(X)} elderly samples from: {data_dir}"
    )

    print("\n" + "=" * 60)
    print("  Fine-tuning complete!")
    print("=" * 60)
    print(f"  Adapter saved → models/fer/elderly_adapter.pkl")
    print(f"  Trained on:    {len(X)} samples")
    print(f"  Classes:       {sorted(set(y))}")
    print("\n  The web UI will now use the elderly-adapted model automatically.")
    print("  Run: .\\venv\\Scripts\\python scripts\\run_fer_web_ui.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
