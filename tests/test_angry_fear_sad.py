"""
Verify Angry, Fear, and Sad geometric activations work correctly.
These are unit tests for the landmark-based boosting logic.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fer.emotion_model import EmotionClassifier

FACE = np.full((64, 64, 3), 128, dtype=np.uint8)


def make_clf_with_baseline(mouth_width, nose_to_mouth):
    """
    Initialize classifier and set baseline geometry explicitly.
    mouth_width and nose_to_mouth are the IOD-normalised resting values
    that will be stored and used as the reference.
    """
    clf = EmotionClassifier()
    clf.baseline_geometry = {
        'mouth_width': float(mouth_width),
        'nose_to_mouth': float(nose_to_mouth)
    }
    return clf


def landmarks_from_geo(mouth_width_iod, nose_to_mouth_iod, iod=100.0):
    """
    Build a landmarks dict where the geometry exactly produces the
    requested IOD-normalized values (using float arithmetic to avoid
    integer-rounding errors).
    """
    cx, ey = 200.0, 100.0
    # Eyes
    re = (cx - iod / 2, ey)
    le = (cx + iod / 2, ey)
    # Nose (arbitrary y, not used in geometry calculation)
    ny = ey + 50.0
    nose = (cx, ny)
    # Mouth: half-width = mouth_width_iod * iod / 2
    # nose_to_mouth_iod = (mouth_cy - ny) / iod  =>  mouth_cy = ny + nose_to_mouth_iod * iod
    mouth_cy = ny + nose_to_mouth_iod * iod
    half_mw = (mouth_width_iod * iod) / 2.0
    rm = (cx - half_mw, mouth_cy)
    lm = (cx + half_mw, mouth_cy)
    return {
        'right_eye': re, 'left_eye': le,
        'nose': nose,
        'right_mouth': rm, 'left_mouth': lm
    }


# Baseline resting values
BASE_MW  = 0.85   # resting mouth_width / IOD
BASE_NTM = 0.65   # resting nose_to_mouth / IOD


def test_neutral_resting():
    """
    Resting face: mouth width and height exactly at baseline => dW=0, dH=0 => is_resting=True.
    """
    clf = make_clf_with_baseline(BASE_MW, BASE_NTM)
    lm = landmarks_from_geo(BASE_MW, BASE_NTM)
    result = clf.predict(FACE, landmarks=lm)
    probs = result['probabilities']
    geo = result['geo_debug']

    print(f"\n[NEUTRAL] dW={geo['dW']:.4f} dH={geo['dH']:.4f} resting={geo['is_resting']}")
    for emo, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {emo:<10} {p:.3f}")

    assert geo['is_resting'], f"dW={geo['dW']:.4f} dH={geo['dH']:.4f} — should be resting"
    assert probs['Neutral'] > probs['Happy'], \
        f"Neutral {probs['Neutral']:.3f} should beat Happy {probs['Happy']:.3f}"
    assert probs['Neutral'] > probs['Angry'], \
        f"Neutral {probs['Neutral']:.3f} should beat Angry {probs['Angry']:.3f}"
    print("  [PASS] Neutral correctly dominates at rest")


def test_angry_activation():
    """
    Pursed / compressed lips: mouth width clearly below baseline.
    dW = -0.25 (well below -0.020 threshold), dH ~ 0.
    """
    clf = make_clf_with_baseline(BASE_MW, BASE_NTM)
    # mouth_width = 0.60 => dW = 0.60 - 0.85 = -0.25
    lm = landmarks_from_geo(mouth_width_iod=0.60, nose_to_mouth_iod=BASE_NTM)
    result = clf.predict(FACE, landmarks=lm)
    probs = result['probabilities']
    geo = result['geo_debug']

    print(f"\n[ANGRY] dW={geo['dW']:.4f} dH={geo['dH']:.4f} resting={geo['is_resting']}")
    for emo, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {emo:<10} {p:.3f}")

    assert not geo['is_resting'], "Should NOT be in resting state"
    assert geo['dW'] < -0.020, f"dW should be below -0.020, got {geo['dW']:.4f}"
    assert probs['Angry'] > probs['Neutral'], \
        f"Angry {probs['Angry']:.3f} should beat Neutral {probs['Neutral']:.3f}"
    assert probs['Angry'] > probs['Happy'], \
        f"Angry {probs['Angry']:.3f} should beat Happy {probs['Happy']:.3f}"
    print("  [PASS] Angry correctly dominates")


def test_fear_activation():
    """
    Jaw drop + horizontal mouth stretch.
    dW = +0.25 (> 0.012), dH = +0.50 (> 0.030) => Fear branch.
    """
    clf = make_clf_with_baseline(BASE_MW, BASE_NTM)
    # mouth_width=1.10 => dW=+0.25, nose_to_mouth=1.15 => dH=+0.50
    lm = landmarks_from_geo(mouth_width_iod=1.10, nose_to_mouth_iod=1.15)
    result = clf.predict(FACE, landmarks=lm)
    probs = result['probabilities']
    geo = result['geo_debug']

    print(f"\n[FEAR] dW={geo['dW']:.4f} dH={geo['dH']:.4f} resting={geo['is_resting']}")
    for emo, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {emo:<10} {p:.3f}")

    assert not geo['is_resting'], "Should NOT be in resting state"
    assert geo['dH'] > 0.030, f"dH should be > 0.030, got {geo['dH']:.4f}"
    assert geo['dW'] > 0.012, f"dW should be > 0.012, got {geo['dW']:.4f}"
    assert probs['Fear'] > probs['Neutral'], \
        f"Fear {probs['Fear']:.3f} should beat Neutral {probs['Neutral']:.3f}"
    assert probs['Fear'] > probs['Happy'], \
        f"Fear {probs['Fear']:.3f} should beat Happy {probs['Happy']:.3f}"
    print("  [PASS] Fear correctly dominates")


def test_sad_activation():
    """
    Drooping mouth corners, mouth stays closed/narrow.
    dW = -0.05 (<= 0.012), dH = +0.50 (> 0.028) => Sad branch.
    """
    clf = make_clf_with_baseline(BASE_MW, BASE_NTM)
    # mouth_width=0.80 => dW=-0.05, nose_to_mouth=1.15 => dH=+0.50
    lm = landmarks_from_geo(mouth_width_iod=0.80, nose_to_mouth_iod=1.15)
    result = clf.predict(FACE, landmarks=lm)
    probs = result['probabilities']
    geo = result['geo_debug']

    print(f"\n[SAD] dW={geo['dW']:.4f} dH={geo['dH']:.4f} resting={geo['is_resting']}")
    for emo, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {emo:<10} {p:.3f}")

    assert not geo['is_resting'], "Should NOT be in resting state"
    assert probs['Sad'] > probs['Happy'], \
        f"Sad {probs['Sad']:.3f} should beat Happy {probs['Happy']:.3f}"
    print("  [PASS] Sad correctly dominates")


def test_happy_activation():
    """
    Genuine smile: dW = +0.25 (> 0.038), dH stays low (<= 0.028).
    """
    clf = make_clf_with_baseline(BASE_MW, BASE_NTM)
    # mouth_width=1.10 => dW=+0.25, nose_to_mouth=0.65 => dH=0
    lm = landmarks_from_geo(mouth_width_iod=1.10, nose_to_mouth_iod=BASE_NTM)
    result = clf.predict(FACE, landmarks=lm)
    probs = result['probabilities']
    geo = result['geo_debug']

    print(f"\n[HAPPY] dW={geo['dW']:.4f} dH={geo['dH']:.4f} resting={geo['is_resting']}")
    for emo, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {emo:<10} {p:.3f}")

    assert not geo['is_resting'], "Should NOT be in resting state"
    assert geo['dW'] > 0.038, f"dW should be > 0.038, got {geo['dW']:.4f}"
    assert probs['Happy'] > probs['Neutral'], \
        f"Happy {probs['Happy']:.3f} should beat Neutral {probs['Neutral']:.3f}"
    print("  [PASS] Happy correctly activates only on genuine smile")


if __name__ == '__main__':
    test_neutral_resting()
    test_angry_activation()
    test_fear_activation()
    test_sad_activation()
    test_happy_activation()
    print("\n All 5 geometric activation tests passed!")
