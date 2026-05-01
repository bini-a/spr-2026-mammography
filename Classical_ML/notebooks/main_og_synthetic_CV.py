
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from scipy.sparse import hstack
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Project-specific imports (unchanged)
from preprocess_pt import *  # stop_words, preprocess_text, tfidf_word, tfidf_char
from ablation_synthetic import best_prep_fn



# CONFIG

RANDOM_STATE = 42
N_SPLITS = 5
OUTPUT_DIR = "outputs_og_synthetic_CV"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paths — edit these two if your data lives elsewhere.
TRAIN_PATH = "/Users/Loyan/vscode/CS534/Project/synth_data/og_synth_train.csv"
TEST_PATH = "test.csv"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})
COLORS = ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30",
          "#7F77DD", "#E24B4A", "#5DCAA5"]
BIRADS_LABELS = [f"BI-RADS {i}" for i in range(7)]



# LOAD DATA AND FLAG SYNTHETIC ROWS

print("Loading and exploring data")
print("-" * 65)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Flag each training row as original (is_synthetic = 0) or synthetic (1).
# Original rows use IDs like "Acc1234"; synthetic rows have purely numeric IDs.
train["is_synthetic"] = (
    ~train["ID"].astype(str).str.startswith("Acc")
).astype(int)

n_orig = int((train["is_synthetic"] == 0).sum())
n_synth = int((train["is_synthetic"] == 1).sum())

print(f"  Train total   : {train.shape[0]:,} rows, {train.shape[1]} columns")
print(f"    Original    : {n_orig:,} rows  (used for both training and validation)")
print(f"    Synthetic   : {n_synth:,} rows  (added to training folds only)")
print(f"  Test          : {test.shape[0]:,} rows, {test.shape[1]} columns")
print(f"  Nulls in train: {train.isnull().sum().sum()}")

# Target distribution (overall + split by source)
print("\n  Target distribution (overall):")
for label, count in train["target"].value_counts().sort_index().items():
    pct = count / len(train) * 100
    print(f"    BI-RADS {label}: {count:>6} ({pct:5.1f}%)")

# target distribution, stacked by source 
target_counts = train["target"].value_counts().sort_index()
orig_counts = (
    train[train["is_synthetic"] == 0]["target"]
    .value_counts().reindex(range(7), fill_value=0)
)
synth_counts = (
    train[train["is_synthetic"] == 1]["target"]
    .value_counts().reindex(range(7), fill_value=0)
)

plt.figure(figsize=(9, 5))
x = np.arange(7)
plt.bar(x, orig_counts.values, color=COLORS[0],
        edgecolor="black", label="Original")
plt.bar(x, synth_counts.values, bottom=orig_counts.values,
        color=COLORS[2], edgecolor="black", label="Synthetic")
for i in range(7):
    total = orig_counts.values[i] + synth_counts.values[i]
    plt.text(i, total * 1.05, f"{total}", ha="center", fontsize=9)
plt.xlabel("BI-RADS Class")
plt.ylabel("Number of reports")
plt.title("Target distribution (training set, stacked by source)")
plt.yscale("log")
plt.xticks(x, BIRADS_LABELS)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/synth_og_plot1_target_distribution.png", dpi=150)
plt.close()

#report-length distribution 
train["report_len"] = train["report"].str.len()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(train["report_len"], bins=50,
             color=COLORS[0], alpha=0.8, edgecolor="white")
axes[0].set_xlabel("Report length (characters)")
axes[0].set_ylabel("Count")
axes[0].set_title("Report length distribution")
axes[0].axvline(
    train["report_len"].median(), color=COLORS[3], linestyle="--",
    label=f'Median: {train["report_len"].median():.0f}',
)
axes[0].legend()

for i in range(7):
    subset = train[train["target"] == i]["report_len"]
    axes[1].hist(subset, bins=30, alpha=0.5,
                 label=f"BI-RADS {i}", color=COLORS[i])
axes[1].set_xlabel("Report length (characters)")
axes[1].set_ylabel("Count")
axes[1].set_title("Report length by BI-RADS class")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/synth_og_plot2_report_lengths.png",
            bbox_inches="tight")
plt.close()


# TEXT PREPROCESSING

print("\nText preprocessing (NLTK stopwords & stemmer, winning recipe from ablation)")
print("-" * 65)
print(f"  Stopwords : {len(stop_words)} Portuguese words")
print(f"  Recipe    : best_prep_fn (from ablation_synthetic)")

# Apply the winning preprocessing once
train["clean"] = train["report"].astype(str).apply(best_prep_fn)
test["clean"] = test["report"].astype(str).apply(best_prep_fn)

# Sample preview
sample_text = train["report"].iloc[0]
print(f"\n  Sample original : {sample_text[:90]}...")
print(f"  Sample cleaned  : {train['clean'].iloc[0][:90]}...")



#  FOLD INDICES 
print("\nBuilding CV folds (stratified on original rows only)")
print("-" * 65)

is_synth = train["is_synthetic"].values.astype(bool)
orig_idx = np.where(~is_synth)[0]   # global indices of original rows
synth_idx = np.where(is_synth)[0]   # global indices of synthetic rows

y = train["target"].values
y_orig = y[orig_idx]

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Pre-compute the 5 folds so every model uses identical splits
folds = []
for tr_local, val_local in skf.split(orig_idx, y_orig):
    tr_orig_global = orig_idx[tr_local]         # original training part
    val_global = orig_idx[val_local]            # original-only validation
    tr_global = np.concatenate([tr_orig_global, synth_idx])  # + all synthetic
    folds.append((tr_global, val_global))

for i, (tr, val) in enumerate(folds, 1):
    n_tr_synth = int(is_synth[tr].sum())
    n_val_synth = int(is_synth[val].sum())
    print(f"  Fold {i}: train={len(tr):>6,} "
          f"({len(tr) - n_tr_synth:,} orig + {n_tr_synth:,} synth)   "
          f"val={len(val):>5,} ({n_val_synth:,} synth — should be 0)")



# MODELS

models = {
    "LogReg": LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Linear SVC": LinearSVC(
        C=1.0, class_weight="balanced", max_iter=2000,
        random_state=RANDOM_STATE,
    ),
    "SGD": SGDClassifier(
        loss="modified_huber", class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "MultiNB": MultinomialNB(alpha=0.1),
    "CompNB": ComplementNB(alpha=0.1),
    "LightGBM": LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    ),
}


# 5-FOLD CV  FIT INSIDE EACH FOLD (no validation leakage)

print("\nModel training (5-fold stratified CV, synthetic rows in training folds only)")
print("-" * 65)

from sklearn.base import clone  # fresh model per fold

model_results = {name: {"fold_macros": [], "fold_weighteds": []}
                 for name in models}
model_cms = {name: np.zeros((7, 7), dtype=int) for name in models}
model_all_preds = {name: ([], []) for name in models}  # (y_true, y_pred)

clean_texts = train["clean"].values

for fold_i, (tr_global, val_global) in enumerate(folds, 1):
    print(f"\n  Fold {fold_i}/{N_SPLITS}")

    # Fit TF-IDF on THIS fold's training text only
    word_vec = clone(tfidf_word)
    char_vec = clone(tfidf_char)
    X_tr_word = word_vec.fit_transform(clean_texts[tr_global])
    X_tr_char = char_vec.fit_transform(clean_texts[tr_global])
    X_tr = hstack([X_tr_word, X_tr_char]).tocsr()

    X_val_word = word_vec.transform(clean_texts[val_global])
    X_val_char = char_vec.transform(clean_texts[val_global])
    X_val = hstack([X_val_word, X_val_char]).tocsr()

    y_tr = y[tr_global]
    y_val = y[val_global]

    for name, model in models.items():
        m = clone(model)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_val)

        model_results[name]["fold_macros"].append(
            f1_score(y_val, pred, average="macro")
        )
        model_results[name]["fold_weighteds"].append(
            f1_score(y_val, pred, average="weighted")
        )
        model_cms[name] += confusion_matrix(y_val, pred, labels=range(7))
        yt, yp = model_all_preds[name]
        yt.extend(y_val); yp.extend(pred)

# Aggregate per-model metrics
model_per_class = {}
for name in models:
    macros = model_results[name]["fold_macros"]
    weighteds = model_results[name]["fold_weighteds"]
    yt, yp = model_all_preds[name]

    prec, rec, f1, sup = precision_recall_fscore_support(
        yt, yp, labels=range(7), average=None
    )

    model_results[name].update({
        "f1_macro": float(np.mean(macros)),
        "f1_macro_std": float(np.std(macros)),
        "f1_weighted": float(np.mean(weighteds)),
        "fold_scores": macros,
    })
    model_per_class[name] = {
        "precision": prec, "recall": rec, "f1": f1, "support": sup,
    }

    print(f"\n  {name}")
    print(f"    F1 Macro   : {np.mean(macros):.4f} ± {np.std(macros):.4f}")
    print(f"    F1 Weighted: {np.mean(weighteds):.4f}")

best_name = max(model_results, key=lambda k: model_results[k]["f1_macro"])
print(f"\n   Best model: {best_name}  "
      f"(F1 macro = {model_results[best_name]['f1_macro']:.4f})")



# model comparison

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_names = list(model_results.keys())
macros = [model_results[n]["f1_macro"] for n in model_names]
weighteds = [model_results[n]["f1_weighted"] for n in model_names]
x_pos = np.arange(len(model_names))
w = 0.35

bars1 = axes[0].bar(x_pos - w / 2, macros, w,
                    label="F1 Macro", color=COLORS[0], edgecolor="white")
bars2 = axes[0].bar(x_pos + w / 2, weighteds, w,
                    label="F1 Weighted", color=COLORS[1], edgecolor="white")
for bars, vals in [(bars1, macros), (bars2, weighteds)]:
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center",
                     fontsize=9, fontweight="bold")

axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(model_names, fontsize=9, rotation=30, ha="right")
axes[0].set_ylabel("F1 Score")
axes[0].set_title("Model comparison (synth rows excluded from validation)")
# Auto y-limit so labels never overflow
axes[0].set_ylim(min(macros + weighteds) - 0.05,
                 max(macros + weighteds) + 0.05)
axes[0].legend()

fold_data = [model_results[n]["fold_scores"] for n in model_names]
bp = axes[1].boxplot(fold_data, labels=model_names,
                     patch_artist=True, widths=0.5)
for patch, color in zip(bp["boxes"], COLORS[:len(model_names)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_xticklabels(model_names, rotation=30, ha="right")
axes[1].set_ylabel("F1 Macro per fold")
axes[1].set_title("Cross-validation stability")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/synth_og_plot4_model_comparison.png",
            bbox_inches="tight")
plt.close()



# DETAILED EVALUATION — best model

print(f"\nDetailed evaluation — {best_name}")
print("-" * 65)

best_cm = model_cms[best_name]
best_pc = model_per_class[best_name]

print(f"\n  Per-class metrics (accumulated over out-of-fold predictions):")
print(f"  {'Class':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
print(f"  {'-' * 50}")
for c in range(7):
    print(
        f"  BI-RADS {c}  {best_pc['precision'][c]:>9.3f}"
        f" {best_pc['recall'][c]:>8.3f}"
        f" {best_pc['f1'][c]:>8.3f}"
        f" {int(best_pc['support'][c]):>8}"
    )

# confusion matrix
fig, ax = plt.subplots(figsize=(8, 7))
row_sums = best_cm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # avoid div-by-zero for empty classes
cm_norm = best_cm.astype(float) / row_sums

sns.heatmap(
    cm_norm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=BIRADS_LABELS, yticklabels=BIRADS_LABELS,
    linewidths=0.5, linecolor="white", ax=ax,
    cbar_kws={"label": "Proportion"}, vmin=0, vmax=1,
)
for i in range(7):
    for j in range(7):
        ax.text(j + 0.5, i + 0.72, f"({best_cm[i, j]})",
                ha="center", va="center", fontsize=7, color="gray")

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix — {best_name}\n"
             "(Normalized by row, raw counts in parentheses)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/synth_og_plot5_confusion_matrix.png",
            bbox_inches="tight")
plt.close()

#per-class precision / recall / F1 
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(7)
w = 0.25
bars_p = ax.bar(x_pos - w, best_pc["precision"], w,
                label="Precision", color=COLORS[0], edgecolor="white")
bars_r = ax.bar(x_pos, best_pc["recall"], w,
                label="Recall", color=COLORS[1], edgecolor="white")
bars_f = ax.bar(x_pos + w, best_pc["f1"], w,
                label="F1 Score", color=COLORS[2], edgecolor="white")
for bar_group in [bars_p, bars_r, bars_f]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(BIRADS_LABELS)
ax.set_ylabel("Score")
ax.set_title(f"Per-class precision, recall & F1 — {best_name}")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right")
for i, s in enumerate(best_pc["support"]):
    ax.text(i, -0.06, f"n={int(s):,}", ha="center", fontsize=8,
            color="gray", transform=ax.get_xaxis_transform())
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/synth_og_plot6_per_class_metrics.png",
            bbox_inches="tight")
plt.close()

# misclassification heatmap 
fig, ax = plt.subplots(figsize=(8, 6))
misclass = cm_norm.copy()
np.fill_diagonal(misclass, 0)
sns.heatmap(
    misclass, annot=True, fmt=".2f", cmap="Reds",
    xticklabels=BIRADS_LABELS, yticklabels=BIRADS_LABELS,
    linewidths=0.5, linecolor="white", ax=ax, vmin=0, vmax=0.3,
    cbar_kws={"label": "Misclassification rate"},
)
ax.set_xlabel("Predicted (incorrectly)")
ax.set_ylabel("Actual class")
ax.set_title("Misclassification patterns")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/synth_og_plot7_misclassification_heatmap.png",
            bbox_inches="tight")
plt.close()



# FINAL MODEL 
print("\nFinal model: fit on ALL training data and predict test set")
print("-" * 65)

# Re-fit TF-IDF on the full training set for the final model
X_train_word = tfidf_word.fit_transform(train["clean"])
X_train_char = tfidf_char.fit_transform(train["clean"])
X_train_full = hstack([X_train_word, X_train_char]).tocsr()

X_test_word = tfidf_word.transform(test["clean"])
X_test_char = tfidf_char.transform(test["clean"])
X_test_full = hstack([X_test_word, X_test_char]).tocsr()

print(f"  Word features : {X_train_word.shape[1]:,}")
print(f"  Char features : {X_train_char.shape[1]:,}")
print(f"  Total features: {X_train_full.shape[1]:,}")

best_model = clone(models[best_name])
best_model.fit(X_train_full, y)

# Feature importance (only for linear models)
if hasattr(best_model, "coef_"):
    print("\nFeature importance analysis")
    print("-" * 65)

    feature_names = (
        tfidf_word.get_feature_names_out().tolist()
        + tfidf_char.get_feature_names_out().tolist()
    )

    n_top = 10
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes_flat = axes.flatten()

    for c in range(7):
        ax = axes_flat[c]
        coefs = best_model.coef_[c]
        top_idx = np.argsort(coefs)[-n_top:]
        top_features = [feature_names[i] for i in top_idx]
        top_values = [coefs[i] for i in top_idx]

        bars = ax.barh(range(n_top), top_values,
                       color=COLORS[c], edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n_top))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel("Coefficient weight")
        ax.set_title(f"BI-RADS {c}", fontweight="bold")
        for bar, val in zip(bars, top_values):
            ax.text(bar.get_width() + 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)

    axes_flat[7].axis("off")
    plt.suptitle(
        f"Top {n_top} discriminative features per BI-RADS class ({best_name})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/synth_og_plot8_feature_importance.png",
                bbox_inches="tight")
    plt.close()

    print("\n  Top 5 features per class:")
    for c in range(7):
        top_idx = np.argsort(best_model.coef_[c])[-5:][::-1]
        feats = ", ".join([feature_names[i] for i in top_idx])
        print(f"    BI-RADS {c}: {feats}")

# Test set predictions
test_preds = best_model.predict(X_test_full)
test["predicted_target"] = test_preds

submission = test[["ID", "predicted_target"]]
sub_path = f"{OUTPUT_DIR}/synth_og_submission.csv"
submission.to_csv(sub_path, index=False)
print(f"\n  Predictions saved to {sub_path}")
print(f"  Prediction distribution: "
      f"{pd.Series(test_preds).value_counts().sort_index().to_dict()}")
