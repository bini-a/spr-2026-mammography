"""
Generate the Kaggle inference notebook for exp035_svc_best.
Blends 4-seed BERTimbau ensemble (55%) with LinearSVC+TF-IDF (45%).

Datasets to attach in the Kaggle notebook:
  1. biniamgaromsa/mammography-exp998-blend-v1  (BERT models — already uploaded)
  2. biniamgaromsa/mammography-svc-model        (model.pkl + vectorizers.pkl — new, ~5 MB)
  3. spr-2026-mammography-report-classification (competition data)

Thresholds are embedded directly — no separate thresholds upload needed.
"""
import json
import os

import numpy as np

BERT_COMPONENTS = [
    "exp023_bertimbau_dedup",
    "exp015a_bertimbau_seed7",
    "exp015b_bertimbau_seed13",
    "exp015c_bertimbau_seed21",
]
BERT_WEIGHT = 0.55
SVC_WEIGHT  = 0.45
OFFSETS     = np.load("experiments/exp035_svc_best/thresholds.npy").tolist()
OUT_NB      = "experiments/exp035_svc_best/notebook_inference.ipynb"

# Use exp998 notebook as template — same 4 BERT components, same writefile cells
with open("experiments/exp998_blend_base/notebook_inference.ipynb") as f:
    nb = json.load(f)

# ---- Patch the markdown header cell (cell 0) ----
nb["cells"][0]["source"] = (
    f"# exp035_svc_best — BERT+SVC hybrid ensemble inference\n\n"
    f"**Architecture:** 4-seed BERTimbau ensemble ({BERT_WEIGHT:.0%}) + "
    f"LinearSVC TF-IDF ({SVC_WEIGHT:.0%})\n"
    f"**BERT components ({len(BERT_COMPONENTS)} models):** {', '.join(BERT_COMPONENTS)}\n\n"
    f"OOF tuned macro-F1: **0.7814** — highest ever (vs 0.7645 for BERT-only exp025).\n"
    "Thresholds embedded — no separate threshold upload needed.\n\n"
    "## Setup checklist\n"
    "1. **Accelerator**: GPU T4×1\n"
    "2. **Internet**: Off\n"
    "3. **Competition dataset**: attach `spr-2026-mammography-report-classification`\n"
    "4. **BERT dataset**: attach `biniamgaromsa/mammography-exp998-blend-v1`\n"
    "   (contains exp023, exp015a, exp015b, exp015c)\n"
    "5. **SVC dataset**: attach `biniamgaromsa/mammography-svc-model`\n"
    "   (contains model.pkl + vectorizers.pkl — ~5 MB total)\n"
)

# ---- Find and replace the inference cell ----
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "COMPONENTS" in src and "ensemble_probs" in src:
        inf_cell_idx = i
        break

offsets_str = repr(OFFSETS)
new_source = f"""\
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import load_test
from src.features import clean_text
from src.train_transformer import _TextDataset

BERT_COMPONENTS = {BERT_COMPONENTS!r}
BERT_WEIGHT     = {BERT_WEIGHT}
SVC_WEIGHT      = {SVC_WEIGHT}
MAX_LENGTH      = 256
CLASSES         = [0, 1, 2, 3, 4, 5, 6]

# Thresholds embedded from experiments/exp035_svc_best/thresholds.npy
OFFSETS = np.array({offsets_str})

# ── Locate BERT dataset ──────────────────────────────────────────────────────
def _find_bert_root():
    for root, dirs, files in os.walk('/kaggle/input'):
        for comp in BERT_COMPONENTS:
            if comp in dirs:
                print(f'Found BERT dataset at: {{root}}')
                return root
    for root, dirs, files in os.walk('/kaggle/input'):
        if BERT_COMPONENTS[0] in root:
            return os.path.dirname(root)
    raise FileNotFoundError(
        'BERT models not found — attach biniamgaromsa/mammography-exp998-blend-v1')

# ── Locate SVC dataset ───────────────────────────────────────────────────────
def _find_svc_dir():
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'model.pkl' in files and 'vectorizers.pkl' in files:
            print(f'Found SVC model at: {{root}}')
            return root
    raise FileNotFoundError(
        'SVC model not found — attach biniamgaromsa/mammography-svc-model')

BERT_ROOT = _find_bert_root()
SVC_DIR   = _find_svc_dir()

# ── Load test data ───────────────────────────────────────────────────────────
test = load_test()
test['text'] = test['report'].apply(clean_text)
print(f'Test rows: {{len(test)}}')
texts = test['text'].tolist()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {{device}}')

# ── SVC inference ────────────────────────────────────────────────────────────
print('\\nRunning SVC inference...')
with open(os.path.join(SVC_DIR, 'vectorizers.pkl'), 'rb') as f:
    vectorizers = pickle.load(f)
with open(os.path.join(SVC_DIR, 'model.pkl'), 'rb') as f:
    svc_model = pickle.load(f)

Xs = [vec.transform(texts) for vec in vectorizers]
X_test = hstack(Xs, format='csr')

raw_probs = svc_model.predict_proba(X_test)
svc_probs = np.zeros((len(texts), len(CLASSES)))
for i, cls in enumerate(svc_model.classes_):
    svc_probs[:, CLASSES.index(int(cls))] = raw_probs[:, i]
print(f'SVC probs shape: {{svc_probs.shape}}')

# ── BERT inference ───────────────────────────────────────────────────────────
def _load_model_via_symlink(src_dir, idx):
    link = f'_mdl{{idx}}'
    if os.path.lexists(link):
        os.unlink(link)
    os.symlink(os.path.abspath(src_dir), os.path.abspath(link))
    try:
        tok   = AutoTokenizer.from_pretrained(link, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            link, local_files_only=True)
    finally:
        os.unlink(link)
    return model, tok

os.chdir('/kaggle/working')
bert_probs = None
n_bert = len(BERT_COMPONENTS)
for i, comp in enumerate(BERT_COMPONENTS):
    w = 1.0 / n_bert  # uniform within BERT ensemble
    model_path = os.path.join(BERT_ROOT, comp)
    if not os.path.isdir(model_path):
        candidates = [d for d, _, fs in os.walk(BERT_ROOT)
                      if 'config.json' in fs and comp in d]
        if not candidates:
            raise FileNotFoundError(f'Model dir for {{comp}} not found under {{BERT_ROOT}}')
        model_path = candidates[0]
    print(f'\\nLoading {{comp}} from {{model_path}} (BERT uniform weight={{w:.3f}})')
    model, tokenizer = _load_model_via_symlink(model_path, i)
    model = model.to(device).eval()

    ds     = _TextDataset(texts, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            ttids = batch.get('token_type_ids')
            kwargs = {{'input_ids': ids, 'attention_mask': mask}}
            if ttids is not None:
                kwargs['token_type_ids'] = ttids.to(device)
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                out = model(**kwargs)
            all_probs.append(torch.softmax(out.logits.float(), dim=-1).cpu().numpy())

    comp_probs = np.concatenate(all_probs, axis=0)
    bert_probs = comp_probs * w if bert_probs is None else bert_probs + comp_probs * w
    del model
    torch.cuda.empty_cache()
    print(f'  done — shape {{comp_probs.shape}}')

print(f'\\nBERT probs shape: {{bert_probs.shape}}')

# ── Blend BERT + SVC ─────────────────────────────────────────────────────────
final_probs = BERT_WEIGHT * bert_probs + SVC_WEIGHT * svc_probs
print(f'\\nBlend: {{BERT_WEIGHT:.0%}} BERT + {{SVC_WEIGHT:.0%}} SVC')

# ── Apply embedded thresholds ─────────────────────────────────────────────────
preds = np.argmax(final_probs + OFFSETS, axis=1)
print(f'Thresholds applied: {{np.round(OFFSETS, 4).tolist()}}')

sub = pd.DataFrame({{'ID': test['ID'], 'target': preds}})
sub.to_csv('submission.csv', index=False)
print(f'\\nsubmission.csv written: {{len(sub)}} rows')
print(sub['target'].value_counts().sort_index().to_string())
"""

nb["cells"][inf_cell_idx]["source"] = new_source

os.makedirs("experiments/exp035_svc_best", exist_ok=True)
with open(OUT_NB, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook written → {OUT_NB}")
print(f"BERT components : {BERT_COMPONENTS}")
print(f"BERT weight     : {BERT_WEIGHT}")
print(f"SVC weight      : {SVC_WEIGHT}")
print(f"Offsets         : {[round(o, 4) for o in OFFSETS]}")
print()
print("Datasets to attach on Kaggle:")
print("  1. biniamgaromsa/mammography-exp998-blend-v1  (BERT models)")
print("  2. biniamgaromsa/mammography-svc-model        (SVC model, ~5 MB — NEW)")
print("  3. spr-2026-mammography-report-classification (competition data)")
