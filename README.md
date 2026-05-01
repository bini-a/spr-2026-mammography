# SPR 2026 Mammography Report Classification

Coursework solution for the [Kaggle SPR 2026 Mammography Report Classification](https://www.kaggle.com/competitions/spr-2026-mammography-report-classification) competition (Emory CS 534 final project).

**Task**: Predict the BI-RADS category (0–6) from the *findings* section of Portuguese-language mammography reports. The *impression* section (which states the BI-RADS category explicitly) has been removed — models must infer the category from descriptive findings alone.

**Metric**: Macro-averaged F1 (`sklearn.metrics.f1_score(..., average='macro')`)

**Final standing**: ranked 21 on final leaderbord

---

## BI-RADS Category Reference

| Class | Meaning |
|-------|---------|
| 0 | Incomplete — needs additional imaging |
| 1 | Negative |
| 2 | Benign |
| 3 | Probably benign |
| 4 | Suspicious |
| 5 | Highly suggestive of malignancy |
| 6 | Known biopsy-proven malignancy |

Class distribution is severely imbalanced: class 2 accounts for ~87% of training examples; classes 5 and 6 have only 29 and 45 examples respectively.

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies (creates .venv automatically)
~/.local/bin/uv sync

# Or via Makefile
make install
```

**Hardware requirements**: GPU strongly recommended. Developed on 2× NVIDIA RTX 6000 Ada (49 GB VRAM), CUDA 12.4. Kaggle T4 GPU works for inference notebooks.

---

## Project Structure

```
src/
  data.py               # load_train(), load_test(), make_folds() — group-aware stratified K-fold
  features.py           # clean_text(), build_features(), transform_features() — TF-IDF
  train.py              # OOF CV loop for sklearn/GBM models
  train_transformer.py  # GPU OOF loop — FP16, R-Drop, AWP, W&B, early stopping
  ensemble.py           # Post-hoc OOF probability averaging + threshold tuning
  rerank.py             # Specialist soft-reranking on top of a base ensemble
  threshold.py          # Per-class decision threshold tuning (random search on OOF)
  evaluate.py           # compute_metrics(), print_metrics(), append_to_results_log()
  predict.py            # Inference dispatch (sklearn or transformer)
  notebook_gen.py       # Kaggle notebook generation utilities
  models/
    linear.py           # LogReg / LinearSVC (sklearn, isotonic calibration)
    gbm.py              # LightGBM
    transformer.py      # HuggingFace save/load (not pickle)

configs/                # One YAML per experiment
scripts/                # Utility and notebook-generation scripts
experiments/            # Output artifacts per run (gitignored: model weights, OOF probs)
notebooks/              # EDA and error analysis
run.py                  # CLI entry point — dispatches to sklearn or transformer loop
Makefile                # Convenience wrappers
```

---

## Running Experiments

```bash
# Train only
uv run python run.py configs/exp023_bertimbau_dedup.yaml --train

# Train + generate submission (requires local test.csv)
uv run python run.py configs/exp023_bertimbau_dedup.yaml

# Compare all experiments by OOF macro-F1
uv run python run.py --compare

# Transformer training (single GPU)
TOKENIZERS_PARALLELISM=false uv run python run.py configs/exp023_bertimbau_dedup.yaml --train
```

**GPU selection**: Set `gpu: 0` or `gpu: 1` in the config YAML. Do not use `CUDA_VISIBLE_DEVICES` alongside the `gpu:` field — they conflict.

### Starting a New Experiment

**sklearn / GBM**:
1. Copy `configs/base.yaml` → `configs/expNNN_description.yaml`
2. Set `experiment_name` and `model.type` (`logistic_regression`, `linear_svc`, or `lgbm`)
3. Run with `--train`

**Transformer**:
1. Copy `configs/base_transformer.yaml` → `configs/expNNN_description.yaml`
2. Set `experiment_name` and `model.pretrained` (HuggingFace model ID or local path)
3. Optionally enable W&B: `wandb.enabled: true`
4. Run with `TOKENIZERS_PARALLELISM=false uv run python run.py configs/expNNN.yaml --train`

### Ensemble and Reranking

```bash
# Average OOF probs across experiments, tune thresholds
uv run python -m src.ensemble exp023_bertimbau_dedup exp015a_bertimbau_seed7 \
  exp015b_bertimbau_seed13 exp015c_bertimbau_seed21 \
  --out exp025_multiseed_ensemble --n-iter 2000

# Specialist soft-rerank on top of a base
uv run python -m src.rerank \
  --base exp025_multiseed_ensemble \
  --spec023 exp034a_023_specialist \
  --spec456 exp040_specialist_456_synth \
  --out exp044_rerank_a023_0.3_a456_0.1 \
  --alpha023 0.3 --alpha456 0.1 --n-iter 2000
```

---

## Key Findings

### What works
- **BERTimbau base** (Portuguese-native BERT) outperforms all multilingual models (XLM-R, mDeBERTa, BioBERTpt) on this dataset.
- **Early stopping** (patience=2, up to 10 epochs) prevents overfitting and yields consistent gains vs fixed epoch counts.
- **Multi-seed ensembling** (seeds 42, 7, 13, 21) adds +0.021 OOF macro-F1 over a single model.
- **LinearSVC + TF-IDF** (char+word n-grams) is a surprisingly strong baseline (0.718 OOF) and complements BERT well in blends — particularly for class 5 (+0.074 from SVC diversity).
- **Synthetic rare-class augmentation** (classes 0, 3, 4, 5, 6) adds +0.015 OOF vs the honest BERTimbau baseline.
- **Specialist reranking**: training separate models on confusion groups {0,2,3} and {4,5,6} then soft-blending on gated rows adds +0.009 OOF.
- **Per-class threshold tuning** on OOF probabilities (random search, 2000 iterations) is allowed by competition rules and is a reliable macro-F1 lever.
- **R-Drop and AWP**: regularization techniques for rare-class calibration (exp045, exp046 — results pending).

### What does not work
- **Focal loss**: collapses rare-class F1 (−0.058 vs baseline). Conflicts with class-weighted CE.
- **Label smoothing**: collapses class 5 F1 from 0.567 → 0.208 (exp011).
- **LLRD (layer-wise LR decay)**: −0.023 OOF on BERTimbau. With only 9k training examples all layers need full LR.
- **WeightedRandomSampler**: −0.068 OOF. Double-penalises the majority signal when combined with class-weighted CE.
- **DAPT (domain-adaptive pre-training)**: 18k reports is too small for MLM pre-training to add signal.
- **LightGBM on raw TF-IDF**: tree models can't exploit linear signal distributed across 300k sparse dimensions.
- **BioBERTpt**: consistently trails BERTimbau; different vocabulary register from clinical SciELO pretraining.

---

## Experiment Results

| Experiment | OOF Macro-F1 | Notes |
|-----------|-------------|-------|
| exp044_rerank_a023_0.3_a456_0.1 | **0.7878** | 5-seed BERT+SVC + specialist rerank — best OOF |
| exp035_svc_best | 0.7814 | 4-seed BERT (55%) + SVC (45%) — **LB 0.80678** |
| exp034_rerank (α023=0.4, α456=0.2) | 0.7732 | Specialist rerank on 4-seed ensemble |
| exp025_multiseed_ensemble | 0.7645 | 4-seed BERTimbau — **LB 0.80578** |
| exp030_bertimbau_synthetic_rare | 0.7581 | BERTimbau + synthetic augmentation {0,3,4,5,6} |
| exp023_bertimbau_dedup | 0.743 | Honest single-model baseline (group-aware K-fold) |
| exp002_tfidf_svc | 0.718 | Best sklearn baseline |

*All macro-F1 values are post-threshold-tuning. Experiments before exp023 used leaky folds (inflated ~0.022).*

---

## Kaggle Submission Workflow

**Inference notebooks** (recommended — load pre-trained weights, ~5 min vs 2–3h training):

1. Upload model weights as a Kaggle dataset (`kaggle datasets create`).
2. Generate the inference notebook:
   ```bash
   uv run python scripts/gen_exp044_notebook.py
   ```
3. Import notebook on Kaggle: New Notebook → File → Import → upload `.ipynb`.
4. Attach datasets: model weights + competition data.
5. Settings: Accelerator = GPU T4×1, Internet = Off.
6. Save & Run All → Submit.

---

