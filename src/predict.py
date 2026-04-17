"""
Generate submission.csv from a trained experiment directory.
Called by run.py — do not invoke directly.
"""
import os

import numpy as np
import pandas as pd

from src.data import load_test
from src.features import clean_text, load_vectorizers, transform_features
from src.models.linear import load_model

CLASSES = [0, 1, 2, 3, 4, 5, 6]


def run_predict(exp_dir: str) -> str:
    test = load_test()
    test["text"] = test["report"].apply(clean_text)

    vectorizers = load_vectorizers(exp_dir)
    model = load_model(exp_dir)

    X_test = transform_features(test["text"].tolist(), vectorizers)
    probs = model.predict_proba(X_test)

    # Align probability columns to CLASSES order
    prob_matrix = np.zeros((len(test), len(CLASSES)))
    for i, cls in enumerate(model.classes_):
        prob_matrix[:, CLASSES.index(int(cls))] = probs[:, i]

    # Apply thresholds if they were tuned for this experiment
    threshold_path = os.path.join(exp_dir, "thresholds.npy")
    if os.path.exists(threshold_path):
        offsets = np.load(threshold_path)
        preds = np.argmax(prob_matrix + offsets, axis=1)
        print("Applied saved threshold offsets.")
    else:
        preds = np.argmax(prob_matrix, axis=1)

    sub = pd.DataFrame({"ID": test["ID"], "target": preds})
    sub_path = os.path.join(exp_dir, "submission.csv")
    sub.to_csv(sub_path, index=False)

    print(f"Submission saved → {sub_path}  ({len(sub)} rows)")
    print(f"Prediction distribution:\n{sub['target'].value_counts().sort_index().to_string()}")
    return sub_path
