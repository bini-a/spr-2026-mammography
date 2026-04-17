"""
Post-processing: per-class decision threshold tuning on OOF probabilities.
Tuning is explicitly allowed by competition rules. Always tune on OOF only.
"""
import numpy as np
from sklearn.metrics import f1_score

CLASSES = [0, 1, 2, 3, 4, 5, 6]


def tune_thresholds(
    oof_probs: np.ndarray,
    oof_labels: np.ndarray,
    n_iter: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Random-search for per-class additive offsets that maximise OOF macro F1.
    Returns an offset array of shape (n_classes,).
    Apply at inference time: preds = argmax(probs + offsets, axis=1).
    """
    baseline_preds = np.argmax(oof_probs, axis=1)
    best_f1 = f1_score(
        oof_labels, baseline_preds, average="macro", labels=CLASSES, zero_division=0
    )
    best_offsets = np.zeros(len(CLASSES))
    print(f"Threshold tuning baseline macro F1: {best_f1:.4f}")

    rng = np.random.RandomState(seed)
    for _ in range(n_iter):
        offsets = rng.uniform(-0.3, 0.3, size=len(CLASSES))
        preds = np.argmax(oof_probs + offsets, axis=1)
        f1 = f1_score(oof_labels, preds, average="macro", labels=CLASSES, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_offsets = offsets.copy()

    print(f"Threshold tuning best macro F1:     {best_f1:.4f}")
    print(f"Offsets: {dict(zip(CLASSES, best_offsets.round(4)))}")
    return best_offsets


def apply_thresholds(probs: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.argmax(probs + offsets, axis=1)
