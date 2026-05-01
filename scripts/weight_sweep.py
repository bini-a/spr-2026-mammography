"""
Random weight sweep over available OOF files.
Finds optimal blend weights to maximise tuned macro-F1.

Usage:
    uv run python scripts/weight_sweep.py
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.threshold import tune_thresholds

CLASSES = [0, 1, 2, 3, 4, 5, 6]

EXPS = [
    "exp023_bertimbau_dedup",
    "exp015a_bertimbau_seed7",
    "exp015b_bertimbau_seed13",
    "exp015c_bertimbau_seed21",
]

N_ITER          = 100_000  # weight combinations (raw F1 only — fast)
TUNE_ITER_FINAL = 2000     # threshold tuning only for top candidates
SEED            = 42
TOP_K           = 20       # re-evaluate top-K with full threshold tuning


def load_oofs():
    all_probs, labels_ref = [], None
    for exp in EXPS:
        path = os.path.join("experiments", exp, "oof_preds.csv")
        df   = pd.read_csv(path)
        prob_cols = [f"p{c}" for c in CLASSES]
        probs  = df[prob_cols].values.astype(float)
        labels = df["target"].values.astype(int)
        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise ValueError(f"Label mismatch in {exp}")
        all_probs.append(probs)
        print(f"  loaded {exp}  ({len(probs):,} rows)")
    return all_probs, labels_ref


def blend(all_probs, weights):
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    return sum(p * wi for p, wi in zip(all_probs, w)), w


def raw_f1(all_probs, weights, labels):
    avg, _ = blend(all_probs, weights)
    return f1_score(labels, avg.argmax(axis=1), average="macro")


def evaluate_full(all_probs, weights, labels, n_iter=TUNE_ITER_FINAL, seed=SEED):
    avg, w = blend(all_probs, weights)
    offsets  = tune_thresholds(avg, labels, n_iter=n_iter, seed=seed)
    tuned_f1 = f1_score(labels, (avg + offsets).argmax(axis=1), average="macro")
    return tuned_f1, offsets, w


def main():
    rng = np.random.default_rng(SEED)
    print(f"\nLoading OOF files...")
    all_probs, labels = load_oofs()
    n = len(EXPS)

    # ---- Baseline: uniform 4-seed ----
    f1_025, _, _ = evaluate_full(all_probs, [1, 1, 1, 1], labels)
    print(f"\nBaseline exp025 (uniform 4-seed):  tuned OOF = {f1_025:.4f}")

    # ---- Phase 1: raw-F1 sweep (fast) ----
    print(f"\nPhase 1: sweeping {N_ITER:,} weight combos on raw argmax F1...")
    t0 = time.time()
    results = []
    for i in range(N_ITER):
        w = rng.dirichlet(alpha=np.ones(n))
        f1 = raw_f1(all_probs, w, labels)
        results.append((f1, w.tolist()))
        if (i + 1) % 20000 == 0:
            best_so_far = max(r[0] for r in results)
            print(f"  {i+1:>7,} / {N_ITER:,}  |  best raw F1: {best_so_far:.4f}  ({time.time()-t0:.0f}s)")

    results.sort(reverse=True)
    top = results[:TOP_K]
    print(f"Phase 1 done in {time.time()-t0:.0f}s  |  best raw F1: {results[0][0]:.4f}")

    # ---- Phase 2: full threshold tuning on top-K ----
    print(f"\nPhase 2: threshold-tuning top {TOP_K} configs ({TUNE_ITER_FINAL} iters each)...")
    final_results = []
    for rank, (approx_f1, w) in enumerate(top, 1):
        final_f1, offsets, norm_w = evaluate_full(all_probs, w, labels)
        final_results.append((final_f1, norm_w, offsets))
        w_str = "  ".join(f"{wi:.3f}" for wi in norm_w)
        print(f"  {rank:<3}  tuned={final_f1:.4f}  raw={approx_f1:.4f}  [{w_str}]")

    print(f"\n{'='*65}")
    print(f"TOP {TOP_K} WEIGHT CONFIGURATIONS (tuned F1)")
    print(f"{'='*65}")
    print(f"{'Rank':<5} {'Tuned F1':>9}  Weights (exp023, 015a, 015b, 015c)")
    print(f"{'-'*65}")
    final_results.sort(reverse=True)
    for rank, (f1, norm_w, _) in enumerate(final_results, 1):
        w_str = "  ".join(f"{wi:.3f}" for wi in norm_w)
        print(f"  {rank:<3}  {f1:.4f}   [{w_str}]")

    best_f1, best_w, best_offsets = final_results[0]

    print(f"\n{'='*65}")
    print(f"BEST CONFIG")
    print(f"{'='*65}")
    print(f"  Tuned OOF macro-F1 : {best_f1:.4f}")
    print(f"  vs exp025 baseline  : {best_f1 - f1_025:+.4f}")
    print(f"\n  Weights:")
    for exp, w in zip(EXPS, best_w):
        print(f"    {exp:<40}  {w:.4f}")

    print(f"\n  Ensemble command:")
    w_args = " ".join(f"{w:.4f}" for w in best_w)
    exp_args = " ".join(EXPS)
    print(f"    uv run python -m src.ensemble {exp_args} \\")
    print(f"        --weights {w_args} \\")
    print(f"        --out exp999_final_ensemble --n-iter 2000")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
