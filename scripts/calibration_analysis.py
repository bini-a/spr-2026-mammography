"""
Calibration analysis for the best ensemble model (exp025 / exp998_blend_base).

Computes:
  - Per-class reliability diagrams (confidence vs fraction positive)
  - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
  - Brier score (overall and per class)
  - Temperature scaling: fits T on OOF log-loss, reports calibrated ECE

Usage:
    uv run python scripts/calibration_analysis.py
    uv run python scripts/calibration_analysis.py --exp exp998_blend_base
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import brier_score_loss, f1_score, log_loss

CLASSES = [0, 1, 2, 3, 4, 5, 6]
CLASS_NAMES = ["c0 Incomplete", "c1 Negative", "c2 Benign",
               "c3 Prob. Benign", "c4 Suspicious", "c5 Highly Susp.", "c6 Known Malig."]
N_BINS = 10


# --------------------------------------------------------------------------- #
# Calibration helpers
# --------------------------------------------------------------------------- #

def reliability_data(probs, labels, n_bins=N_BINS):
    """Return (bin_centers, frac_pos, mean_conf, bin_counts) for a binary problem."""
    bins = np.linspace(0, 1, n_bins + 1)
    centers, frac_pos, mean_conf, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        frac_pos.append(labels[mask].mean())
        mean_conf.append(probs[mask].mean())
        counts.append(mask.sum())
    return np.array(centers), np.array(frac_pos), np.array(mean_conf), np.array(counts)


def ece(probs, labels, n_bins=N_BINS):
    """Expected Calibration Error (weighted mean |conf - acc| over bins)."""
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(probs)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece_val += mask.sum() / n * abs(conf - acc)
    return ece_val


def mce(probs, labels, n_bins=N_BINS):
    """Maximum Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    mce_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        mce_val = max(mce_val, abs(conf - acc))
    return mce_val


def multiclass_ece(prob_matrix, labels, n_bins=N_BINS):
    """
    ECE for a multi-class problem: uses the confidence of the predicted class
    (top-1 confidence calibration).
    """
    top_conf  = prob_matrix.max(axis=1)
    top_class = prob_matrix.argmax(axis=1)
    correct   = (top_class == labels).astype(float)
    return ece(top_conf, correct, n_bins)


def multiclass_mce(prob_matrix, labels, n_bins=N_BINS):
    top_conf  = prob_matrix.max(axis=1)
    top_class = prob_matrix.argmax(axis=1)
    correct   = (top_class == labels).astype(float)
    return mce(top_conf, correct, n_bins)


def fit_temperature(logits, labels):
    """
    Fit a single temperature scalar T > 0 that minimises NLL on the given logits.
    Returns optimal T.
    """
    def nll(log_t):
        t = np.exp(log_t)   # keep T > 0
        scaled = logits / t
        return log_loss(labels, softmax(scaled, axis=1))

    result = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    T = np.exp(result.x)
    return T


def to_logits(probs, eps=1e-7):
    """Invert softmax approximately: logit = log(p) (up to a constant shift)."""
    return np.log(np.clip(probs, eps, 1 - eps))


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #

def analyse(exp_name: str):
    path = os.path.join("experiments", exp_name, "oof_preds.csv")
    df   = pd.read_csv(path)

    prob_cols = [f"p{c}" for c in CLASSES]
    probs  = df[prob_cols].values.astype(float)
    labels = df["target"].values.astype(int)

    preds  = probs.argmax(axis=1)
    macro_f1 = f1_score(labels, preds, average="macro")

    print(f"\n{'='*60}")
    print(f"  Calibration Analysis: {exp_name}")
    print(f"  OOF macro-F1 (raw argmax): {macro_f1:.4f}")
    print(f"  N = {len(labels):,}  |  classes: {sorted(np.unique(labels).tolist())}")
    print(f"{'='*60}")

    # ---- Multi-class top-1 calibration ------------------------------------ #
    mc_ece = multiclass_ece(probs, labels)
    mc_mce = multiclass_mce(probs, labels)
    nll    = log_loss(labels, probs)
    print(f"\n[Multi-class top-1 calibration]")
    print(f"  ECE  : {mc_ece:.4f}")
    print(f"  MCE  : {mc_mce:.4f}")
    print(f"  NLL  : {nll:.4f}")

    # ---- Per-class (one-vs-rest) calibration ------------------------------ #
    print(f"\n[Per-class ECE / MCE / Brier (one-vs-rest)]")
    print(f"  {'Class':<20}  {'N':>5}  {'ECE':>6}  {'MCE':>6}  {'Brier':>6}  {'Overconf':>9}")
    per_class_ece = {}
    for c, name in zip(CLASSES, CLASS_NAMES):
        y_bin = (labels == c).astype(float)
        p_c   = probs[:, c]
        c_ece = ece(p_c, y_bin)
        c_mce = mce(p_c, y_bin)
        c_bri = brier_score_loss(y_bin, p_c)
        mean_conf = p_c.mean()
        base_rate = y_bin.mean()
        overconf  = mean_conf - base_rate
        per_class_ece[c] = c_ece
        print(f"  {name:<20}  {int(y_bin.sum()):>5}  {c_ece:>6.4f}  {c_mce:>6.4f}  {c_bri:>6.4f}  {overconf:>+9.4f}")

    # ---- Temperature scaling ---------------------------------------------- #
    print(f"\n[Temperature Scaling]")
    logits = to_logits(probs)
    T = fit_temperature(logits, labels)
    scaled_probs = softmax(logits / T, axis=1)

    mc_ece_cal = multiclass_ece(scaled_probs, labels)
    mc_mce_cal = multiclass_mce(scaled_probs, labels)
    nll_cal    = log_loss(labels, scaled_probs)
    preds_cal  = scaled_probs.argmax(axis=1)
    f1_cal     = f1_score(labels, preds_cal, average="macro")

    print(f"  Optimal T : {T:.4f}")
    print(f"  ECE before / after : {mc_ece:.4f} → {mc_ece_cal:.4f}  (Δ {mc_ece_cal - mc_ece:+.4f})")
    print(f"  MCE before / after : {mc_mce:.4f} → {mc_mce_cal:.4f}  (Δ {mc_mce_cal - mc_mce:+.4f})")
    print(f"  NLL before / after : {nll:.4f} → {nll_cal:.4f}  (Δ {nll_cal - nll:+.4f})")
    print(f"  Macro-F1 before / after : {macro_f1:.4f} → {f1_cal:.4f}  (Δ {f1_cal - macro_f1:+.4f})")

    print(f"\n  Per-class ECE after temperature scaling:")
    print(f"  {'Class':<20}  {'ECE before':>10}  {'ECE after':>10}  {'Δ ECE':>8}")
    for c, name in zip(CLASSES, CLASS_NAMES):
        y_bin = (labels == c).astype(float)
        p_c_cal = scaled_probs[:, c]
        c_ece_cal = ece(p_c_cal, y_bin)
        print(f"  {name:<20}  {per_class_ece[c]:>10.4f}  {c_ece_cal:>10.4f}  {c_ece_cal - per_class_ece[c]:>+8.4f}")

    # ---- Reliability diagrams -------------------------------------------- #
    _plot_reliability(exp_name, probs, scaled_probs, labels, T)

    return T, mc_ece, mc_ece_cal


def _plot_reliability(exp_name, probs, scaled_probs, labels, T):
    out_dir = os.path.join("experiments", exp_name)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    # Multi-class top-1 diagram
    ax = axes[0]
    top_conf  = probs.max(axis=1)
    top_class = probs.argmax(axis=1)
    correct   = (top_class == labels).astype(float)
    _, frac_pos, mean_conf, counts = reliability_data(top_conf, correct)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.scatter(mean_conf, frac_pos, s=counts / counts.max() * 300,
               color="steelblue", zorder=3, label="Before T-scaling")

    # Overlay T-scaled
    top_conf_cal = scaled_probs.max(axis=1)
    top_class_cal = scaled_probs.argmax(axis=1)
    correct_cal   = (top_class_cal == labels).astype(float)
    _, fp_cal, mc_cal, cnt_cal = reliability_data(top_conf_cal, correct_cal)
    ax.scatter(mc_cal, fp_cal, s=cnt_cal / cnt_cal.max() * 300,
               color="crimson", marker="^", zorder=3, label=f"T={T:.2f}")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("Top-1 confidence (all classes)")
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Per-class reliability diagrams
    for idx, (c, name) in enumerate(zip(CLASSES, CLASS_NAMES)):
        ax = axes[idx + 1]
        y_bin = (labels == c).astype(float)
        _, fp, mc, cnt = reliability_data(probs[:, c], y_bin)
        _, fp_cal, mc_cal, cnt_cal = reliability_data(scaled_probs[:, c], y_bin)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        if len(mc) > 0:
            ax.scatter(mc, fp, s=np.sqrt(cnt) * 8,
                       color="steelblue", zorder=3, label="Before")
        if len(mc_cal) > 0:
            ax.scatter(mc_cal, fp_cal, s=np.sqrt(cnt_cal) * 8,
                       color="crimson", marker="^", zorder=3, label=f"T={T:.2f}")
        ax.set_title(f"{name}\n(N={int(y_bin.sum())})")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Frac. positive")
        ax.legend(fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.suptitle(f"Reliability diagrams — {exp_name}", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "calibration_reliability.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Reliability diagrams saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp025_multiseed_ensemble",
                        help="Experiment name (default: exp025_multiseed_ensemble)")
    args = parser.parse_args()
    analyse(args.exp)
