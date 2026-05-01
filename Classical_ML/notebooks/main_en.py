import pandas as pd
import numpy as np
import re
import warnings
import os
warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess_en import *
from ablation_en import best_prep_fn

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from scipy.sparse import hstack
# from tqdm import tqdm
# tqdm.pandas()

# NLTK built-in English stopwords and stemmer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# CONFIG

RANDOM_STATE = 42
N_SPLITS = 5
OUTPUT_DIR = "outputs_en"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
COLORS = [
    "#378ADD", "#1D9E75", "#EF9F27", "#D85A30",
    "#7F77DD", "#E24B4A", "#5DCAA5",
]
BIRADS_LABELS = [f"BI-RADS {i}" for i in range(7)]



#Loading and exploring data

print("-" * 65)
print("Load translated data ")


train = pd.read_csv("train_translated.csv")
test = pd.read_csv("test_translated.csv")


print(f"  Train: {train.shape[0]:,} samples, {train.shape[1]} columns")
print(f"  Test : {test.shape[0]} samples, {test.shape[1]} columns")
print(f"  Nulls: {train.isnull().sum().sum()}")
print(f"  Columns       : {train.columns.tolist()}")



# EXPLORE DATA

print("\n" + "-" * 65)
print("Explore translated data")


target_counts = train["target"].value_counts().sort_index()
for label, count in target_counts.items():
    print(f"    BI-RADS {label}: {count:>6} ({count / len(train) * 100:5.1f}%)")

#Target distribution
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar([f"BI-RADS {i}" for i in target_counts.index], target_counts.values,
              color=COLORS, edgecolor="white")
for bar, count in zip(bars, target_counts.values):
    pct = count / len(train) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("BI-RADS Class")
ax.set_ylabel("Number of reports")
ax.set_title("Target distribution (English)")
ax.set_yscale("log") 
ax.set_ylim(10, 30000)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/en_plot1_target_distribution.png", bbox_inches="tight")
plt.show()
plt.close()


#length distribution 
train["report_len"] = train["report_en"].str.len()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(train["report_len"], bins=50, color=COLORS[0], alpha=0.8, edgecolor="white")
axes[0].set_xlabel("Characters")
axes[0].set_title("English report lengths")
axes[0].axvline(train["report_len"].median(), color=COLORS[3], linestyle="--",
                label=f'Median: {train["report_len"].median():.0f}') 
axes[0].legend()

for i in range(7):
    axes[1].hist(train[train["target"] == i]["report_len"], bins=30, alpha=0.5,
                 label=f"BI-RADS {i}", color=COLORS[i])
axes[1].set_xlabel("Report length (characters)")
axes[1].set_title("Report length by BI-RADS class")
axes[1].set_ylabel("Count")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/en_plot2_report_lengths.png", bbox_inches="tight")
plt.show()
plt.close()


# Text preprocessing

print("\n" + "-" * 65)
print("Text preprocessing (English NLTK stopwords & stemmer)")
#print("=" * 65)

stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

print(f"  Stopwords : {len(stop_words)} English words (nltk.corpus.stopwords)")
print(f"  Stemmer   : SnowballStemmer('english') (nltk.stem)")

print(f"\n  Preprocessing demonstration:")
print("-" * 65)

sample_text = train["report_en"].iloc[0]
print(f"Original: {sample_text[:100]}...")
print(f"Clean Only: {preprocess_text(sample_text, remove_stops=False, stem=False, apply_lemma=False)[:100]}...")
print(f"Clean + No Stopwords: {preprocess_text(sample_text, remove_stops=True, stem=False, apply_lemma=False)[:100]}...")
print(f"Clean + Stemming: {preprocess_text(sample_text, remove_stops=False, stem=True, apply_lemma=False)[:100]}...")
print(f"Clean + Lemmatization: {preprocess_text(sample_text, remove_stops=False, stem=False, apply_lemma=True)[:100]}...")


#apply best preprocessing from ablation
#PREPROCESSING ABLATION

y = train["target"].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



# Best preprocessing from ablation study
train["clean"] = train["report_en"].apply(best_prep_fn)
test["clean"] = test["report_en"].apply(best_prep_fn)
# print(f"  Applied to train and test sets.")


# TF-IDF FEATURES

print("\n" + "-" * 65)
print("TF-IDF feature engineering (word + char n-grams)")


# tfidf_word = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
# tfidf_char = TfidfVectorizer(analyzer="char_wb", max_features=10000, ngram_range=(3, 5), min_df=3, max_df=0.95, sublinear_tf=True)

X_word = tfidf_word.fit_transform(train["clean"])
X_char = tfidf_char.fit_transform(train["clean"])
X_train = hstack([X_word, X_char])
X_test = hstack([tfidf_word.transform(test["clean"]), tfidf_char.transform(test["clean"])])
print(f"  Word features: {X_word.shape[1]:,} | Character features: {X_char.shape[1]:,} | Total features: {X_train.shape[1]:,}")


#  MODEL TRAINING

print("\n" + "-" * 65)
print("Model training (5-fold stratified CV)")
print("-" * 65)

models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Linear SVC": LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),

    "SGD Classifier": SGDClassifier(
        loss="modified_huber", class_weight="balanced", random_state=RANDOM_STATE,
    ),
    
    "Multinomial NB": MultinomialNB(alpha=0.1),
    "Complement NB": ComplementNB(alpha=0.1),

    "LightGBM": LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    ),

}


# model_results, model_cms, model_per_class = {}, {}, {}
model_results = {}
model_cms = {}
model_per_class = {}

for name, model in models.items():
    fold_m, fold_w = [], []
    cm_total = np.zeros((7, 7), dtype=int)
    all_yt, all_yp = [], []
    for train_idx, val_idx in skf.split(X_train, y):
        model.fit(X_train[train_idx], y[train_idx])
        pred = model.predict(X_train[val_idx])
        fold_m.append(f1_score(y[val_idx], pred, average="macro"))
        fold_w.append(f1_score(y[val_idx], pred, average="weighted"))
        cm_total += confusion_matrix(y[val_idx], pred, labels=range(7))
        all_yt.extend(y[val_idx]); all_yp.extend(pred)

    prec, rec, f1, sup = precision_recall_fscore_support(all_yt, all_yp, labels=range(7), average=None)
    model_results[name] = {"f1_macro": np.mean(fold_m), "f1_macro_std": np.std(fold_m),
                           "f1_weighted": np.mean(fold_w), "fold_scores": fold_m}
    model_cms[name] = cm_total
    model_per_class[name] = {"precision": prec, "recall": rec, "f1": f1, "support": sup}
    print(f"\n  {name}:")
    print(f"    F1 Macro   : {np.mean(fold_m):.4f} ± {np.std(fold_m):.4f}")
    print(f"    F1 Weighted: {np.mean(fold_w):.4f}")

best_name = max(model_results, key=lambda k: model_results[k]["f1_macro"])
print(f"\n   Best model: {best_name} ")

# Model comparison 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
model_names = list(model_results.keys())
x_pos = np.arange(len(model_names))
w = 0.35
macros = [model_results[n]["f1_macro"] for n in model_names]
weighteds = [model_results[n]["f1_weighted"] for n in model_names]

b1 = axes[0].bar(x_pos - w / 2, macros, w, label="F1 Macro", color=COLORS[0], edgecolor="white")
b2 = axes[0].bar(x_pos + w / 2, weighteds, w, label="F1 Weighted", color=COLORS[1], edgecolor="white")
for bar, val in zip(b1, macros):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
for bar, val in zip(b2, weighteds):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(model_names, fontsize=9, rotation=30, ha="right")
axes[0].set_ylabel("F1 Score")
axes[0].set_title("Model comparison (English)")
axes[0].set_ylim(0.4, 1.05)
axes[0].legend()

bp = axes[1].boxplot([model_results[n]["fold_scores"] for n in model_names], labels=model_names, patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], [COLORS[0], COLORS[1], COLORS[2]]): 
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
axes[1].set_xticklabels(model_names, rotation=30, ha="right")
axes[1].set_ylabel("F1 Macro per fold")
axes[1].set_title("Cross-validation stability (English)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/en_plot4_model_comparison.png", bbox_inches="tight")
plt.show()
plt.close()

# DETAILED EVALUATION
print("\n" + "-" * 65)
print(f"Detailed evaluation — {best_name}")
print("-" * 65)

best_cm = model_cms[best_name]
best_pc = model_per_class[best_name]

print(f"\n  Per-class metrics:")
print(f"\n  {'Class':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
print(f"  {'-' * 48}")
for c in range(7):
    print(f"  BI-RADS {c}  {best_pc['precision'][c]:>9.3f} {best_pc['recall'][c]:>8.3f} {best_pc['f1'][c]:>8.3f} {int(best_pc['support'][c]):>8}")

# Confusion matrix
fig, ax = plt.subplots(figsize=(8, 7))
cm_norm = best_cm.astype(float) / best_cm.sum(axis=1, keepdims=True)

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=BIRADS_LABELS, yticklabels=BIRADS_LABELS,
            linewidths=0.5, linecolor="white", ax=ax, vmin=0, vmax=1)
for i in range(7):
    for j in range(7):
        ax.text(j + 0.5, i + 0.72, f"({best_cm[i, j]})", ha="center", va="center", fontsize=7, color="gray")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix — {best_name} (English)\n(Normalized by row, raw counts in parentheses)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/en_plot5_confusion_matrix.png", bbox_inches="tight")
plt.show()
plt.close()

# Per-class metrics
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(7)
w = 0.25

bars_p = ax.bar(x_pos - w, best_pc["precision"], w, label="Precision", color=COLORS[0], edgecolor="white")
bars_r = ax.bar(x_pos, best_pc["recall"], w, label="Recall", color=COLORS[1], edgecolor="white")
bars_f = ax.bar(x_pos + w, best_pc["f1"], w, label="F1", color=COLORS[2], edgecolor="white")

for bar_group in [bars_p, bars_r, bars_f]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x_pos) 
ax.set_xticklabels(BIRADS_LABELS)
ax.set_ylabel("Score")
ax.set_title(f"Per-class metrics - {best_name} (English)")
ax.set_ylim(0, 1.15)
ax.legend()
for i, s in enumerate(best_pc["support"]):
    ax.text(i, -0.06, f"\n n={int(s):,}", ha="center", fontsize=8, color="gray", transform=ax.get_xaxis_transform())
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/en_plot6_per_class_metrics.png", bbox_inches="tight")
plt.show()
plt.close()

# Misclassification
fig, ax = plt.subplots(figsize=(8, 6)); misclass = cm_norm.copy(); np.fill_diagonal(misclass, 0)
sns.heatmap(misclass, annot=True, fmt=".2f", cmap="Reds", xticklabels=BIRADS_LABELS, yticklabels=BIRADS_LABELS,
            linewidths=0.5, linecolor="white", ax=ax, vmin=0, vmax=0.3)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Misclassification patterns (English)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/en_plot7_misclassification_heatmap.png", bbox_inches="tight")
plt.show()
plt.close()


# FEATURE IMPORTANCE
print("Feature importance analysis")
print("-" * 65)

best_model = models[best_name]; best_model.fit(X_train, y)
if hasattr(best_model, "coef_"):
    feature_names = tfidf_word.get_feature_names_out().tolist() + tfidf_char.get_feature_names_out().tolist()
    n_top = 10
    fig, axes = plt.subplots(4, 2, figsize=(14, 16)); axes_flat = axes.flatten()
    for c in range(7):
        ax = axes_flat[c]; coefs = best_model.coef_[c]; top_idx = np.argsort(coefs)[-n_top:]
        axes_flat[c].barh(range(n_top), [coefs[i] for i in top_idx], color=COLORS[c], edgecolor="white")
        axes_flat[c].set_yticks(range(n_top))
        axes_flat[c].set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
        axes_flat[c].set_xlabel("Weight"); axes_flat[c].set_title(f"BI-RADS {c}", fontweight="bold")
    axes_flat[7].axis("off")
    plt.suptitle(f"Top features per class - English ({best_name})", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/en_plot8_feature_importance.png", bbox_inches="tight")
    plt.show()
    plt.close()
    
    print(f"\n  Top 5 features per class:")
    for c in range(7):
        top_idx = np.argsort(best_model.coef_[c])[-5:][::-1]
        print(f"    BI-RADS {c}: {', '.join([feature_names[i] for i in top_idx])}")


#TEST PREDICTIONS for the kaggle submission
print("STEP 9: Test set predictions")
print("-" * 65)

test["predicted_target"] = best_model.predict(X_test)
print(test[["ID", "predicted_target"]].to_string(index=False))
test[["ID", "predicted_target"]].to_csv(f"{OUTPUT_DIR}/en_submission.csv", index=False)




