import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

from preprocess_pt import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from scipy.sparse import hstack


OUTPUT_DIR = "outputs_og_synthetic_CV"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#loading data
train = pd.read_csv("/Users/Loyan/vscode/CS534/Project/synth_data/synth_train.csv")
test = pd.read_csv("/Users/Loyan/vscode/CS534/Project/synth_data/synth_test.csv")

train = train.reset_index(drop=True)
test  = test.reset_index(drop=True)

# Preprocess sample
sample_text = train["report"].iloc[0]
print(f"Original: {sample_text[:100]}...")
print(f"Clean Only: {preprocess_text(sample_text, remove_stopwords=False, apply_stemming=False, apply_lemma=False)[:100]}...")
print(f"Clean + No Stopwords: {preprocess_text(sample_text, remove_stopwords=True, apply_stemming=False, apply_lemma=False)[:100]}...")
print(f"Clean + Stemming: {preprocess_text(sample_text, remove_stopwords=False, apply_stemming=True, apply_lemma=False)[:100]}...")
print(f"Clean + Lemmatization: {preprocess_text(sample_text, remove_stopwords=False, apply_stemming=False, apply_lemma=True)[:100]}...")


#apply best preprocessing (clean + no stopwords) for ablation
train["clean_report"] = train["report"].apply(lambda x: preprocess_text(x, remove_stopwords=True, apply_stemming=False, apply_lemma=False))
test["clean_report"] = test["report"].apply(lambda x: preprocess_text(x, remove_stopwords=True, apply_stemming=False, apply_lemma=False))

# Ablation: Compare different preprocessing steps
y = train["target"].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



ablation_variants = {
    "Baseline\n(Clean only)": lambda x: preprocess_text(x, remove_stopwords=False, apply_stemming=False, apply_lemma=False),
    "Clean + No Stopwords": lambda x: preprocess_text(x, remove_stopwords=True, apply_stemming=False, apply_lemma=False),
    "Clean + \nStemming": lambda x: preprocess_text(x, remove_stopwords=False, apply_stemming=True, apply_lemma=False),
    "Clean + \nLemmatization": lambda x: preprocess_text(x, remove_stopwords=False, apply_stemming=False, apply_lemma=True),
    "Clean + \nNo Stopwords + \nStemming": lambda x: preprocess_text(x, remove_stopwords=True, apply_stemming=True, apply_lemma=False),
    "Clean + \nNo Stopwords + \nLemmatization": lambda x: preprocess_text(x, remove_stopwords=True, apply_stemming=False, apply_lemma=True),
}

ablation_results = {}
    # for variant_name, preprocess_fn in ablation_variants.items():
    #     print(f"\n=== Ablation: {variant_name} ===")
    #     X_variant = train["report"].apply(preprocess_fn)
    #     X_word = tfidf_word.fit_transform(X_variant)
    #     X_char = tfidf_char.fit_transform(X_variant)
    #     X_combined = hstack([X_word, X_char])

    #     macro_scores = []
    #     for train_idx, val_idx in skf.split(X_combined, y):
    #         X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
    #         y_tr, y_val = y[train_idx], y[val_idx]

    #         #model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    #         model = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    #         model.fit(X_tr, y_tr)
    #         y_pred = model.predict(X_val)

    #         macro_scores.append(f1_score(y_val, y_pred, average="macro"))

    #     avg_macro_f1 = np.mean(macro_scores)
    #     ablation_results[variant_name] = avg_macro_f1
    #     print(f"  Average Macro F1: {avg_macro_f1:.4f}")
for name, fn in ablation_variants.items():
    texts = train["report"].apply(fn)
    # tfidf = TfidfVectorizer(
    #     max_features=15000, ngram_range=(1, 2),
    #     min_df=3, max_df=0.95, sublinear_tf=True,
    # )
    X = tfidf_word.fit_transform(texts)
    model = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)

    fold_macros = []
    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        fold_macros.append(f1_score(y[val_idx], pred, average="macro"))

    ablation_results[name] = {
        "mean": np.mean(fold_macros),
        "std": np.std(fold_macros),
        "folds": fold_macros,
    }
    label_clean = name.replace("\n", " ")
    print(f"  {label_clean:30s} F1-macro: {np.mean(fold_macros):.4f} ± {np.std(fold_macros):.4f}")



fig, ax = plt.subplots(figsize=(9, 5))
names = list(ablation_results.keys())
means = [ablation_results[n]["mean"] for n in names]
stds = [ablation_results[n]["std"] for n in names]

bar_colors = ["#888780", "#1D9E75", "#378ADD", "#7F77DD"]
bars = ax.bar(
    names, means, yerr=stds, color=bar_colors, edgecolor="white",
    linewidth=0.5, capsize=5, error_kw={"linewidth": 1.2},
)

for bar, m in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
        f"{m:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

ax.set_ylabel("F1 Macro Score")
ax.set_title("Preprocessing ablation study (Linear SVC, 5-fold CV)")
ax.set_ylim(0.70, 0.76)
ax.axhline(y=means[0], color="#888780", linestyle="--", alpha=0.4, label="Baseline")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/synth_og_plot3_preprocessing_ablation.png", bbox_inches="tight")
plt.show()


best_prep_name = max(ablation_results, key=lambda k: ablation_results[k]["mean"])
best_prep_fn = ablation_variants[best_prep_name]
best_prep_label = best_prep_name.replace("\n", " ")
 
print(f"\n   Best preprocessing: {best_prep_label} "
      f"(F1={ablation_results[best_prep_name]['mean']:.4f}) ")