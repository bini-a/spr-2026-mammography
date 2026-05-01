"""
Generate the Kaggle inference notebook for exp044_rerank_a023_0.3_a456_0.1.

Architecture:
  - 5-seed BERTimbau base (50%) + LinearSVC TF-IDF (50%) → base_probs
  - Spec023 (exp034a, 3-class): soft rerank rows where top-2 ∈ {0,2,3}, α=0.3
  - Spec456 (exp040, 7-class full): soft rerank rows where top-2 ∈ {4,5,6}, α=0.1

Datasets to attach on Kaggle (4 total):
  1. biniamgaromsa/mammography-exp998-blend-v1      (exp023/015a/015b/015c)
  2. biniamgaromsa/mammography-exp044-extras         (exp041 + exp034a + exp040 + thresholds.npy)
  3. biniamgaromsa/mammography-svc-model-exp2        (model.pkl + vectorizers.pkl)
  4. spr-2026-mammography-report-classification      (competition data)

OOF tuned macro-F1: 0.7878
"""
import json
import os

import numpy as np

RERANK_EXP  = "exp044_rerank_a023_0.3_a456_0.1"
BERT_COMPONENTS = [
    "exp023_bertimbau_dedup",
    "exp015a_bertimbau_seed7",
    "exp015b_bertimbau_seed13",
    "exp015c_bertimbau_seed21",
    "exp041_bertimbau_seed99",
]
BERT_WEIGHT = 0.5
SVC_WEIGHT  = 0.5
SPEC023     = "exp034a_023_specialist"
SPEC456     = "exp040_specialist_456_synth"
ALPHA023    = 0.3
ALPHA456    = 0.1
OFFSETS     = np.load(f"experiments/{RERANK_EXP}/thresholds.npy").tolist()
OUT_NB      = f"experiments/{RERANK_EXP}/notebook_inference.ipynb"

with open("experiments/exp998_blend_base/notebook_inference.ipynb") as f:
    nb = json.load(f)

# ---- Patch header cell ----
nb["cells"][0]["source"] = (
    f"# {RERANK_EXP} — 5-seed BERT+SVC with specialist rerank\n\n"
    f"**Architecture:** 5-seed BERTimbau ({BERT_WEIGHT:.0%}) + LinearSVC ({SVC_WEIGHT:.0%}) "
    f"→ soft specialist rerank\n"
    f"**Base BERT ({len(BERT_COMPONENTS)} models):** {', '.join(BERT_COMPONENTS)}\n"
    f"**Spec023:** {SPEC023} (α={ALPHA023}) on rows where top-2 ∈ {{0,2,3}}\n"
    f"**Spec456:** {SPEC456} (α={ALPHA456}) on rows where top-2 ∈ {{4,5,6}}\n\n"
    f"OOF tuned macro-F1: **0.7878**\n\n"
    "## Setup checklist\n"
    "1. **Accelerator**: GPU T4×1\n"
    "2. **Internet**: Off\n"
    "3. **Competition dataset**: `spr-2026-mammography-report-classification`\n"
    "4. **BERT base dataset**: `biniamgaromsa/mammography-exp998-blend-v1`\n"
    "   (exp023, exp015a, exp015b, exp015c)\n"
    "5. **Extras dataset**: `biniamgaromsa/mammography-exp044-extras`\n"
    "   (exp041, exp034a_023_specialist, exp040_specialist_456_synth)\n"
    "6. **SVC dataset**: `biniamgaromsa/mammography-svc-model-exp2`\n"
    "   (model.pkl + vectorizers.pkl)\n"
)

# ---- Find and replace inference cell ----
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
SPEC023         = {SPEC023!r}
SPEC456         = {SPEC456!r}
ALPHA023        = {ALPHA023}
ALPHA456        = {ALPHA456}
MAX_LENGTH      = 256
CLASSES         = [0, 1, 2, 3, 4, 5, 6]
N_CLASSES       = 7

# Thresholds embedded from {RERANK_EXP}/thresholds.npy
OFFSETS = np.array({offsets_str})

# ── Model finder ─────────────────────────────────────────────────────────────
def _find_model_dir(name):
    \"\"\"Walk /kaggle/input for a directory named 'name' containing config.json.\"\"\"
    for root, dirs, files in os.walk('/kaggle/input'):
        if os.path.basename(root) == name and 'config.json' in files:
            print(f'  Found {{name}} at {{root}}')
            return root
    # Fallback: any dir with name in path
    for root, dirs, files in os.walk('/kaggle/input'):
        if name in root and 'config.json' in files:
            print(f'  Found {{name}} (fallback) at {{root}}')
            return root
    raise FileNotFoundError(f'Model dir for {{name}} not found under /kaggle/input')

def _find_svc_dir():
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'model.pkl' in files and 'vectorizers.pkl' in files:
            print(f'  Found SVC model at {{root}}')
            return root
    raise FileNotFoundError('SVC model not found — attach mammography-svc-model-exp2')

def _load_model(model_dir, idx):
    link = f'_mdl{{idx}}'
    if os.path.lexists(link): os.unlink(link)
    os.symlink(os.path.abspath(model_dir), os.path.abspath(link))
    try:
        tok = AutoTokenizer.from_pretrained(link, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(link, local_files_only=True)
    finally:
        os.unlink(link)
    return mdl, tok

def _run_inference(model, tokenizer, texts, device):
    ds = _TextDataset(texts, tokenizer, MAX_LENGTH)
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
    return np.concatenate(all_probs, axis=0)

def _renorm_group(probs, group):
    sub = probs[:, group].copy()
    denom = sub.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return sub / denom

def _blend_group(base_probs, spec_probs, group, alpha):
    out = base_probs.copy()
    base_mass = base_probs[:, group].sum(axis=1, keepdims=True)
    blended = (1 - alpha) * _renorm_group(base_probs, group) + alpha * _renorm_group(spec_probs, group)
    out[:, group] = blended * base_mass
    return out

os.chdir('/kaggle/working')

# ── Load test data ────────────────────────────────────────────────────────────
test = load_test()
test['text'] = test['report'].apply(clean_text)
print(f'Test rows: {{len(test)}}')
texts = test['text'].tolist()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {{device}}')

# ── SVC inference ─────────────────────────────────────────────────────────────
print('\\nRunning SVC inference...')
svc_dir = _find_svc_dir()
with open(os.path.join(svc_dir, 'vectorizers.pkl'), 'rb') as f:
    vectorizers = pickle.load(f)
with open(os.path.join(svc_dir, 'model.pkl'), 'rb') as f:
    svc_model = pickle.load(f)
Xs = [vec.transform(texts) for vec in vectorizers]
X_test = hstack(Xs, format='csr')
raw_probs = svc_model.predict_proba(X_test)
svc_probs = np.zeros((len(texts), N_CLASSES))
for i, cls in enumerate(svc_model.classes_):
    svc_probs[:, CLASSES.index(int(cls))] = raw_probs[:, i]
print(f'SVC probs shape: {{svc_probs.shape}}')

# ── BERT base inference (5 seeds, uniform weight) ─────────────────────────────
print('\\nRunning BERT base inference...')
bert_base_probs = None
n_bert = len(BERT_COMPONENTS)
for i, comp in enumerate(BERT_COMPONENTS):
    w = 1.0 / n_bert
    print(f'  Loading {{comp}} (weight={{w:.3f}})')
    mdl, tok = _load_model(_find_model_dir(comp), i)
    mdl = mdl.to(device).eval()
    p = _run_inference(mdl, tok, texts, device)
    bert_base_probs = p * w if bert_base_probs is None else bert_base_probs + p * w
    del mdl; torch.cuda.empty_cache()
    print(f'  done — shape {{p.shape}}')

# ── Blend BERT + SVC → base_probs ────────────────────────────────────────────
base_probs = BERT_WEIGHT * bert_base_probs + SVC_WEIGHT * svc_probs
print(f'\\nBase probs shape: {{base_probs.shape}}')
print(f'Blend: {{BERT_WEIGHT:.0%}} BERT + {{SVC_WEIGHT:.0%}} SVC')

# ── Specialist rerank ─────────────────────────────────────────────────────────
probs = base_probs.copy()

# Spec023: exp034a (3-class {0,2,3} specialist)
print(f'\\nLoading spec023: {{SPEC023}}')
mdl, tok = _load_model(_find_model_dir(SPEC023), n_bert)
mdl = mdl.to(device).eval()
spec023_raw = _run_inference(mdl, tok, texts, device)
del mdl; torch.cuda.empty_cache()
# exp034a is 3-class: local label order matches training label_subset=[0,2,3]
SPEC023_L2G = {{0: 0, 1: 2, 2: 3}}
spec023_probs = np.zeros((len(texts), N_CLASSES))
for local_idx, global_idx in SPEC023_L2G.items():
    spec023_probs[:, global_idx] = spec023_raw[:, local_idx]
top2 = np.argsort(probs, axis=1)[:, -2:]
gate023 = np.isin(top2, [0, 2, 3]).all(axis=1)
print(f'  gate023 activates on {{gate023.sum()}} rows (α={{ALPHA023}})')
blended023 = _blend_group(probs, spec023_probs, [0, 2, 3], ALPHA023)
probs[gate023] = blended023[gate023]

# Spec456: exp040 (7-class full model, use p4/p5/p6 columns only)
print(f'\\nLoading spec456: {{SPEC456}}')
mdl, tok = _load_model(_find_model_dir(SPEC456), n_bert + 1)
mdl = mdl.to(device).eval()
spec456_probs = _run_inference(mdl, tok, texts, device)  # full 7-class output
del mdl; torch.cuda.empty_cache()
top2 = np.argsort(probs, axis=1)[:, -2:]
gate456 = np.isin(top2, [4, 5, 6]).all(axis=1)
print(f'  gate456 activates on {{gate456.sum()}} rows (α={{ALPHA456}})')
blended456 = _blend_group(probs, spec456_probs, [4, 5, 6], ALPHA456)
probs[gate456] = blended456[gate456]

# ── Apply embedded thresholds ─────────────────────────────────────────────────
preds = np.argmax(probs + OFFSETS, axis=1)
print(f'\\nThresholds applied: {{np.round(OFFSETS, 4).tolist()}}')

sub = pd.DataFrame({{'ID': test['ID'], 'target': preds}})
sub.to_csv('submission.csv', index=False)
print(f'submission.csv written: {{len(sub)}} rows')
print(sub['target'].value_counts().sort_index().to_string())
"""

nb["cells"][inf_cell_idx]["source"] = new_source

os.makedirs(f"experiments/{RERANK_EXP}", exist_ok=True)
with open(OUT_NB, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook written → {OUT_NB}")
print(f"BERT base models : {BERT_COMPONENTS}")
print(f"BERT weight      : {BERT_WEIGHT}")
print(f"SVC weight       : {SVC_WEIGHT}")
print(f"Spec023          : {SPEC023}  α={ALPHA023}")
print(f"Spec456          : {SPEC456}  α={ALPHA456}")
print(f"Offsets          : {[round(o, 4) for o in OFFSETS]}")
print()
print("Datasets to attach on Kaggle (4 datasets):")
print("  1. biniamgaromsa/mammography-exp998-blend-v1   (exp023/015a/015b/015c)")
print("  2. biniamgaromsa/mammography-exp044-extras      (exp041 + exp034a + exp040)")
print("  3. biniamgaromsa/mammography-svc-model-exp2    (model.pkl + vectorizers.pkl)")
print("  4. spr-2026-mammography-report-classification  (competition data)")
print()
print("Upload script for extras dataset:")
print("  bash scripts/prep_exp044_upload.sh")
