import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

from preprocess_en import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report



os.makedirs("outputs_en", exist_ok=True)
    
#loading data
train = pd.read_csv("train_translated.csv")
test = pd.read_csv("test_translated.csv")

# Preprocess sample
sample_text = train["report_en"].iloc[0]
print(f"Original: {sample_text[:100]}...")
print(f"Clean Only: {preprocess_text(sample_text, remove_stops=False, stem=False, apply_lemma=False)[:100]}...")
print(f"Clean + No Stopwords: {preprocess_text(sample_text, remove_stops=True, stem=False, apply_lemma=False)[:100]}...")
print(f"Clean + Stemming: {preprocess_text(sample_text, remove_stops=False, stem=True, apply_lemma=False)[:100]}...")
print(f"Clean + Lemmatization: {preprocess_text(sample_text, remove_stops=False, stem=False, apply_lemma=True)[:100]}...")

#apply best preprocessing (clean + no stopwords) for ablation
train["clean_report"] = train["report_en"].apply(lambda x: preprocess_text(x, remove_stops=True, stem=False, apply_lemma=False))
test["clean_report"] = test["report_en"].apply(lambda x: preprocess_text(x, remove_stops=True, stem=False, apply_lemma=False))


#PREPROCESSING ABLATION

print("\n" + "-" * 65)
print("Preprocessing ablation study")
print("-" * 65)

y = train["target"].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ablation_variants = {
    "Baseline\n(Clean only)": lambda t: preprocess_text(t, remove_stops=False, stem=False, apply_lemma=False),
    "Clean + No Stopwords": lambda t: preprocess_text(t, remove_stops=True, stem=False, apply_lemma=False),
    "Clean + \nStemming": lambda t: preprocess_text(t, remove_stops=False, stem=True, apply_lemma=False),
    "Clean + \nLemmatization": lambda t: preprocess_text(t, remove_stops=False, stem=False, apply_lemma=True),
    "Clean + \nNo Stopwords + \nStemming": lambda t: preprocess_text(t, remove_stops=True, stem=True, apply_lemma=False),
    "Clean + \nNo Stopwords + \nLemmatization": lambda t: preprocess_text(t, remove_stops=True, stem=False, apply_lemma=True),
}
ablation_results = {}
for name, fn in ablation_variants.items():
    texts = train["report_en"].apply(fn)
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                            min_df=3, max_df=0.95, sublinear_tf=True)
    X = tfidf.fit_transform(texts)
    model = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    fold_macros = []
    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        fold_macros.append(f1_score(y[val_idx], pred, average="macro"))
    ablation_results[name] = {"mean": np.mean(fold_macros), "std": np.std(fold_macros), "folds": fold_macros}
    print(f"  {name.replace(chr(10), ' '):30s} F1-macro: {np.mean(fold_macros):.4f} ± {np.std(fold_macros):.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
names = list(ablation_results.keys())
means = [ablation_results[n]["mean"] for n in names]
stds = [ablation_results[n]["std"] for n in names]
bars = ax.bar(names, means, yerr=stds, color=["#888780", "#1D9E75", "#378ADD", "#7F77DD"],
              edgecolor="white", capsize=5)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{m:.4f}", ha="center", fontsize=10)
ax.set_ylabel("F1 Macro") 
ax.set_title("Preprocessing ablation (English, Linear SVC)")
ax.set_ylim(0.70, 0.76)
ax.axhline(y=means[0], color="#888780", linestyle="--", alpha=0.4, label="Baseline") 
ax.legend()
plt.tight_layout()
plt.savefig(f"outputs_en/en_plot3_preprocessing_ablation.png", bbox_inches="tight")
plt.show()
plt.close()


# Auto-select best preprocessing 
best_prep_name = max(ablation_results, key=lambda k: ablation_results[k]["mean"])
best_prep_fn = ablation_variants[best_prep_name]
best_prep_label = best_prep_name.replace("\n", " ")
print(f"\n  Best preprocessing: {best_prep_label} (F1={ablation_results[best_prep_name]['mean']:.4f}) ")



