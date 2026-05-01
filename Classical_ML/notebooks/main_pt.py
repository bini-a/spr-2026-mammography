
import pandas as pd
import numpy as np
import re
import warnings
import os
warnings.filterwarnings("ignore")


import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess_pt import *
from ablation_pt import best_prep_fn

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

# NLTK built-in stopwords and stemmer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer



# CONFIG

RANDOM_STATE = 42
N_SPLITS = 5
os.makedirs("outputs_pt", exist_ok=True)

# Plotting style
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
COLORS = ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30", "#7F77DD", "#E24B4A", "#5DCAA5"]
BIRADS_LABELS = [
    "BI-RADS 0", "BI-RADS 1", "BI-RADS 2", "BI-RADS 3",
    "BI-RADS 4", "BI-RADS 5", "BI-RADS 6",
]


# Loading and exploring data

print("Loading and exploring data")
print("-" * 65)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(f"  Train: {train.shape[0]:,} samples, {train.shape[1]} columns")
print(f"  Test : {test.shape[0]} samples, {test.shape[1]} columns")
print(f"  Nulls: {train.isnull().sum().sum()}")
print(f"  Columns       : {train.columns.tolist()}")

target_counts = train["target"].value_counts().sort_index()
print(f"\n  Target distribution:")
for label, count in target_counts.items():
    pct = count / len(train) * 100
    print(f"    BI-RADS {label}: {count:>6} ({pct:5.1f}%)")

#Target distribution
plt.figure(figsize=(8, 5))
bars = plt.bar(target_counts.index, target_counts.values, color=["dodgerblue", "teal", "orange","orangered",
                                               "mediumslateblue","indianred", "mediumaquamarine" ], edgecolor="black")
for bar, count in zip(bars, target_counts.values):
    pct = count / len(train) * 100
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
             f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
plt.xlabel("BI-RADS Class")
plt.ylabel("Number of reports")
plt.title("Target Distribution (Training Set)")
plt.yscale("log")
plt.xticks(target_counts.index, [f"BI-RADS {i}" for i in target_counts.index])
plt.tight_layout()
plt.savefig("outputs_pt/plot1_target_distribution.png", dpi=150)
plt.show()


#Report length distribution 
train["report_len"] = train["report"].str.len()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(train["report_len"], bins=50, color=COLORS[0], alpha=0.8, edgecolor="white")
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
    axes[1].hist(subset, bins=30, alpha=0.5, label=f"BI-RADS {i}", color=COLORS[i])
axes[1].set_xlabel("Report length (characters)")
axes[1].set_ylabel("Count")
axes[1].set_title("Report length by BI-RADS class")
axes[1].legend(fontsize=8)

plt.tight_layout()

plt.savefig("outputs_pt/plot2_report_lengths.png", bbox_inches="tight")
plt.show()



#Text preprocessing
print("Text preprocessing (NLTK built-in stopwords & stemmer)")
print("-" * 65)

#text resources
print(f"  Stopwords : {len(stop_words)} Portuguese words (nltk.corpus.stopwords)")
print(f"  Stemmer   : SnowballStemmer('portuguese') (nltk.stem)")
print(f"  Lemmatizer  : spaCy lookup lemmatizer (spacy.blank('pt'))")
print(f"  Sample stopwords: {sorted(list(stop_words))[:8]}")


# Sample preprocessing 
print(f"\n  Preprocessing demonstration:")
sample_text = train["report"].iloc[0]
print(f"Original: {sample_text[:100]}...")
print(f"Clean Only: {preprocess_text(sample_text, remove_stopwords=False, apply_stemming=False, apply_lemma=False)[:100]}...")
print(f"Clean + No Stopwords: {preprocess_text(sample_text, remove_stopwords=True, apply_stemming=False, apply_lemma=False)[:100]}...")
print(f"Clean + Stemming: {preprocess_text(sample_text, remove_stopwords=False, apply_stemming=True, apply_lemma=False)[:100]}...")
print(f"Clean + Lemmatization: {preprocess_text(sample_text, remove_stopwords=False, apply_stemming=False, apply_lemma=True)[:100]}...")

# Apply  preprocessing to train and test
# train["clean"] = train["report"].apply(
#     lambda x: preprocess_text(x, remove_stopwords=True, apply_stemming=False, apply_lemma=False)
# )
# test["clean"] = test["report"].apply(
#     lambda x: preprocess_text(x, remove_stopwords=True, apply_stemming=False, apply_lemma=False)
# )


# Preprocessing ablation study
# print("Preprocessing ablation study")
# print("-" * 65)

y = train["target"].values
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


# Apply the winning preprocessing from ablation study to train and test
train["clean"] = train["report"].apply(best_prep_fn)
test["clean"] = test["report"].apply(best_prep_fn)

#TF-IDF feature engineering (word + char n-grams
print("TF-IDF feature engineering (word + char n-grams)")
print("-" * 65)


# train["clean"] = train["report"].apply(lambda x: preprocess_text(x, remove_stopwords=False, apply_stemming=True))
# test["clean"] = test["report"].apply(lambda x: preprocess_text(x, remove_stopwords=False, apply_stemming=True))

# Apply the winning preprocessing to train and test
train["clean_ab"] = train["report"].apply(best_prep_fn)
test["clean_ab"] = test["report"].apply(best_prep_fn)

X_word = tfidf_word.fit_transform(train["clean_ab"])
X_char = tfidf_char.fit_transform(train["clean_ab"])
X_train = hstack([X_word, X_char])

X_test_word = tfidf_word.transform(test["clean_ab"])
X_test_char = tfidf_char.transform(test["clean_ab"])
X_test = hstack([X_test_word, X_test_char])

print(f"  Word features : {X_word.shape[1]:,}")
print(f"  Char features : {X_char.shape[1]:,}")
print(f"  Total features: {X_train.shape[1]:,}")


# Model training (5-fold stratified CV)
print("Model training (5-fold stratified CV)")
print("-" * 65)

# models = {
#     "Logistic Regression": LogisticRegression(
#         C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
#     ),
#     "Linear SVC": LinearSVC(
#         C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
#     ),
#     "Multinomial NB": MultinomialNB(alpha=0.1),
# }

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
model_results = {}
model_cms = {}
model_per_class = {}

for name, model in models.items():
    fold_macros, fold_weighteds = [], []
    cm_total = np.zeros((7, 7), dtype=int)
    all_y_true, all_y_pred = [], []

    for train_idx, val_idx in skf.split(X_train, y):
        model.fit(X_train[train_idx], y[train_idx])
        pred = model.predict(X_train[val_idx])
        fold_macros.append(f1_score(y[val_idx], pred, average="macro"))
        fold_weighteds.append(f1_score(y[val_idx], pred, average="weighted"))
        cm_total += confusion_matrix(y[val_idx], pred, labels=range(7))
        all_y_true.extend(y[val_idx])
        all_y_pred.extend(pred)

    prec, rec, f1, sup = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=range(7), average=None,
    )

    model_results[name] = {
        "f1_macro": np.mean(fold_macros),
        "f1_macro_std": np.std(fold_macros),
        "f1_weighted": np.mean(fold_weighteds),
        "fold_scores": fold_macros,
    }
    model_cms[name] = cm_total
    model_per_class[name] = {"precision": prec, "recall": rec, "f1": f1, "support": sup}

    print(f"\n  {name}:")
    print(f"    F1 Macro   : {np.mean(fold_macros):.4f} ± {np.std(fold_macros):.4f}")
    print(f"    F1 Weighted: {np.mean(fold_weighteds):.4f}")

best_name = max(model_results, key=lambda k: model_results[k]["f1_macro"])
print(f"\n   Best model: {best_name} ")

#Model comparison 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

model_names = list(model_results.keys())
macros = [model_results[n]["f1_macro"] for n in model_names]
weighteds = [model_results[n]["f1_weighted"] for n in model_names]
x_pos = np.arange(len(model_names))
w = 0.35

bars1 = axes[0].bar(x_pos - w / 2, macros, w, label="F1 Macro", color=COLORS[0], edgecolor="white")
bars2 = axes[0].bar(x_pos + w / 2, weighteds, w, label="F1 Weighted", color=COLORS[1], edgecolor="white")
for bar, val in zip(bars1, macros):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
for bar, val in zip(bars2, weighteds):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(model_names, fontsize=9, rotation=30, ha="right")
axes[0].set_ylabel("F1 Score")
axes[0].set_title("Model comparison")
axes[0].set_ylim(0.4, 1.05)
axes[0].legend()

fold_data = [model_results[n]["fold_scores"] for n in model_names]
bp = axes[1].boxplot(fold_data, labels=model_names, patch_artist=True, widths=0.5)
for patch, color in zip(bp["boxes"], [COLORS[0], COLORS[1], COLORS[2]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_xticklabels(model_names, rotation=30, ha="right")
axes[1].set_ylabel("F1 Macro per fold")
axes[1].set_title("Cross-validation stability")
plt.tight_layout()
plt.savefig("outputs_pt/plot4_model_comparison.png", bbox_inches="tight")
plt.show()
plt.close()



# Detailed evaluation
print(f"Detailed evaluation — {best_name}")
print("-" * 65)

best_cm = model_cms[best_name]
best_pc = model_per_class[best_name]

print(f"\n  Per-class metrics:")
print(f"  {'Class':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
print(f"  {'-' * 48}")
for c in range(7):
    print(
        f"  BI-RADS {c}  {best_pc['precision'][c]:>9.3f}"
        f" {best_pc['recall'][c]:>8.3f}"
        f" {best_pc['f1'][c]:>8.3f}"
        f" {int(best_pc['support'][c]):>8}"
    )

# Confusion matrix heatmap 
fig, ax = plt.subplots(figsize=(8, 7))
cm_norm = best_cm.astype(float) / best_cm.sum(axis=1, keepdims=True)

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
ax.set_title(f"Confusion matrix — {best_name}\n(Normalized by row, raw counts in parentheses)")
plt.tight_layout()
plt.savefig("outputs_pt/plot5_confusion_matrix.png", bbox_inches="tight")
plt.show()
plt.close()


#Per-class precision / recall / F1 
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(7)
w = 0.25

bars_p = ax.bar(x_pos - w, best_pc["precision"], w, label="Precision", color=COLORS[0], edgecolor="white")
bars_r = ax.bar(x_pos, best_pc["recall"], w, label="Recall", color=COLORS[1], edgecolor="white")
bars_f = ax.bar(x_pos + w, best_pc["f1"], w, label="F1 Score", color=COLORS[2], edgecolor="white")

for bar_group in [bars_p, bars_r, bars_f]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(BIRADS_LABELS)
ax.set_ylabel("Score")
ax.set_title(f"Per-class precision, recall & F1 - {best_name}")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right")
for i, s in enumerate(best_pc["support"]):
    ax.text(i, -0.06, f"n={int(s):,}", ha="center", fontsize=8, color="gray",
            transform=ax.get_xaxis_transform())
plt.tight_layout()
plt.savefig("outputs_pt/plot6_per_class_metrics.png", bbox_inches="tight")
plt.show()
plt.close()


# Misclassification heatmap 
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
ax.set_title("Misclassification patterns — where does the model get confused?")
plt.tight_layout()
plt.savefig("outputs_pt/plot7_misclassification_heatmap.png", bbox_inches="tight")
plt.show()
plt.close()



# Feature importance analysis
print("Feature importance analysis")
print("-" * 65)

best_model = models[best_name]
best_model.fit(X_train, y)

if hasattr(best_model, "coef_"):
    feature_names = (
        tfidf_word.get_feature_names_out().tolist()
        + tfidf_char.get_feature_names_out().tolist()
    )

    # Top features per class 
    n_top = 10
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes_flat = axes.flatten()

    for c in range(7):
        ax = axes_flat[c]
        coefs = best_model.coef_[c]
        top_idx = np.argsort(coefs)[-n_top:]
        top_features = [feature_names[i] for i in top_idx]
        top_values = [coefs[i] for i in top_idx]

        bars = ax.barh(range(n_top), top_values, color=COLORS[c], edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n_top))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel("SVC coefficient weight")
        ax.set_title(f"BI-RADS {c}", fontweight="bold")

        for bar, val in zip(bars, top_values):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)

    axes_flat[7].axis("off")
    axes_flat[7].text(
        0.1, 0.7,
        "Feature importance summary:\n\n"
        "• BI-RADS 2 (benign): 'benignas', 'esparsas'\n"
        "• BI-RADS 4 (suspicious): 'amorfas', 'pleomórficas'\n"
        "• BI-RADS 5 (malignant): 'espiculado', 'retração'\n"
        "• BI-RADS 6 (confirmed): 'carcinoma', 'invasivo'\n\n"
        "The model learns clinically meaningful\n"
        "patterns from the radiology vocabulary.",
        transform=axes_flat[7].transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8),
    )

    plt.suptitle(
        f"Top {n_top} discriminative features per BI-RADS class ({best_name})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("outputs_pt/plot8_feature_importance.png", bbox_inches="tight")
    plt.show()
    plt.close()
    

    print(f"\n  Top 5 features per class:")
    for c in range(7):
        top_idx = np.argsort(best_model.coef_[c])[-5:][::-1]
        feats = ", ".join([feature_names[i] for i in top_idx])
        print(f"    BI-RADS {c}: {feats}")


# Test set predictions
print("Test set predictions")
print("-" * 65)

test_preds = best_model.predict(X_test)
test["predicted_target"] = test_preds

print(f"\n  Predictions:")
print(test[["ID", "predicted_target"]].to_string(index=False))

submission = test[["ID", "predicted_target"]]
submission.to_csv("outputs_pt/submission.csv", index=False)




