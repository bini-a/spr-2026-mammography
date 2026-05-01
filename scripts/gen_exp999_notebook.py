"""
Generate the Kaggle inference notebook for exp999_final_ensemble.
Reuses models from the already-uploaded exp998 dataset (biniamgaromsa/mammography-exp998-blend-v1).
Thresholds are embedded directly — no new dataset upload needed.
"""
import json, os, numpy as np

COMPONENTS = [
    "exp023_bertimbau_dedup",
    "exp015a_bertimbau_seed7",
    "exp015b_bertimbau_seed13",
    "exp015c_bertimbau_seed21",
]
WEIGHTS = [0.0065, 0.3348, 0.2652, 0.3935]
OFFSETS = np.load("experiments/exp999_final_ensemble/thresholds.npy").tolist()
OUT_NB  = "experiments/exp999_final_ensemble/notebook_inference.ipynb"

# Read the exp998 notebook as a template (already has all source cells)
with open("experiments/exp998_blend_base/notebook_inference.ipynb") as f:
    nb = json.load(f)

# ---- Patch the markdown header cell ----
nb["cells"][0]["source"] = (
    f"# exp999_final_ensemble — optimised 4-seed inference\n\n"
    f"**Components ({len(COMPONENTS)} models):** {', '.join(COMPONENTS)}\n"
    f"**Weights:** {[round(w, 4) for w in WEIGHTS]}\n\n"
    "Optimised 4-seed blend (no synthetic). Thresholds embedded — "
    "attach the existing `biniamgaromsa/mammography-exp998-blend-v1` dataset.\n\n"
    "## Setup checklist\n"
    "1. **Accelerator**: GPU T4×1\n"
    "2. **Internet**: Off\n"
    "3. **Competition dataset**: attach `spr-2026-mammography-report-classification`\n"
    "4. **Model dataset**: attach `biniamgaromsa/mammography-exp998-blend-v1`\n"
    "   (already contains exp023, exp015a, exp015b, exp015c — exp030/031 not used)\n"
)

# ---- Patch the inference cell (last code cell before validation) ----
# Find the cell containing COMPONENTS =
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "COMPONENTS" in src and "WEIGHTS" in src and "ensemble_probs" in src:
        inf_cell_idx = i
        break

offsets_str = repr(OFFSETS)
new_source = f"""\
import os
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data import load_test
from src.features import clean_text
from src.train_transformer import _TextDataset

COMPONENTS   = {COMPONENTS!r}
WEIGHTS      = {WEIGHTS!r}
MAX_LENGTH   = 256

# Thresholds embedded directly from exp999_final_ensemble
OFFSETS = np.array({offsets_str})

# Find the dataset dir containing the model subdirs
def _find_dataset_root():
    for root, dirs, files in os.walk('/kaggle/input'):
        for comp in COMPONENTS:
            if comp in dirs:
                print(f'Found ensemble dataset at: {{root}}')
                return root
    # Fallback: search for any config.json
    for root, dirs, files in os.walk('/kaggle/input'):
        if COMPONENTS[0] in root:
            return os.path.dirname(root)
    raise FileNotFoundError('Model dataset not found — did you attach biniamgaromsa/mammography-exp998-blend-v1?')

DATASET_ROOT = _find_dataset_root()

# Load test data once
test = load_test()
test['text'] = test['report'].apply(clean_text)
print(f'Test rows: {{len(test)}}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {{device}}')

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
ensemble_probs = None
for i, (comp, w) in enumerate(zip(COMPONENTS, WEIGHTS)):
    model_path = os.path.join(DATASET_ROOT, comp)
    if not os.path.isdir(model_path):
        candidates = [d for d, _, fs in os.walk(DATASET_ROOT)
                      if 'config.json' in fs and comp in d]
        if not candidates:
            raise FileNotFoundError(f'Model dir for {{comp}} not found under {{DATASET_ROOT}}')
        model_path = candidates[0]
    print(f'Loading {{comp}} from {{model_path}} (weight={{w:.4f}})')
    model, tokenizer = _load_model_via_symlink(model_path, i)
    model = model.to(device).eval()

    ds     = _TextDataset(test['text'].tolist(), tokenizer, MAX_LENGTH)
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
    ensemble_probs = comp_probs * w if ensemble_probs is None else ensemble_probs + comp_probs * w
    del model
    torch.cuda.empty_cache()
    print(f'  done — shape {{comp_probs.shape}}')

# Apply embedded thresholds
preds = np.argmax(ensemble_probs + OFFSETS, axis=1)
print(f'Thresholds applied: {{OFFSETS.round(4).tolist()}}')

sub = pd.DataFrame({{'ID': test['ID'], 'target': preds}})
sub.to_csv('submission.csv', index=False)
print(f'submission.csv written: {{len(sub)}} rows')
print(sub['target'].value_counts().sort_index().to_string())
"""

nb["cells"][inf_cell_idx]["source"] = new_source

os.makedirs("experiments/exp999_final_ensemble", exist_ok=True)
with open(OUT_NB, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook written → {OUT_NB}")
print(f"Components : {COMPONENTS}")
print(f"Weights    : {[round(w,4) for w in WEIGHTS]}")
print(f"Offsets    : {[round(o,4) for o in OFFSETS]}")
print(f"\nNo new dataset upload needed — reuse biniamgaromsa/mammography-exp998-blend-v1")
