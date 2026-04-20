"""
Generate submission.csv from a trained experiment directory.
Dispatches to sklearn or transformer predict based on what's in the experiment dir.
Called by run.py — do not invoke directly.
"""
import os

import numpy as np
import pandas as pd

from src.data import load_test
from src.features import clean_text, load_vectorizers, transform_features
from src.models import load_model

CLASSES = [0, 1, 2, 3, 4, 5, 6]


def _write_submission(test, preds, exp_dir):
    sub = pd.DataFrame({"ID": test["ID"], "target": preds})
    sub_path = os.path.join(exp_dir, "submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"Submission saved → {sub_path}  ({len(sub)} rows)")
    print(f"Prediction distribution:\n{sub['target'].value_counts().sort_index().to_string()}")
    return sub_path


def run_predict_transformer(exp_dir: str) -> str:
    import torch
    from torch.amp import autocast
    from torch.utils.data import DataLoader
    from src.models.transformer import load_model as load_transformer

    test = load_test()
    test["text"] = test["report"].apply(clean_text)

    model, tokenizer = load_transformer(exp_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Read max_length from saved config if present
    import yaml
    cfg_path = os.path.join(exp_dir, "config.yaml")
    max_length = 512
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        max_length = cfg.get("model", {}).get("max_length", 512)

    from src.train_transformer import _TextDataset
    ds = _TextDataset(test["text"].tolist(), tokenizer, max_length)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            ttids = batch.get("token_type_ids")
            kwargs = {"input_ids": ids, "attention_mask": mask}
            if ttids is not None:
                kwargs["token_type_ids"] = ttids.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                out = model(**kwargs)
            probs = torch.softmax(out.logits.float(), dim=-1).cpu().numpy()
            all_probs.append(probs)

    prob_matrix = np.concatenate(all_probs, axis=0)

    threshold_path = os.path.join(exp_dir, "thresholds.npy")
    if os.path.exists(threshold_path):
        offsets = np.load(threshold_path)
        preds = np.argmax(prob_matrix + offsets, axis=1)
        print("Applied saved threshold offsets.")
    else:
        preds = np.argmax(prob_matrix, axis=1)

    return _write_submission(test, preds, exp_dir)


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

    return _write_submission(test, preds, exp_dir)
