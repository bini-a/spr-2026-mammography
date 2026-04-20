#!/usr/bin/env python3
"""
Domain-Adaptive Pretraining (DAPT): continue MLM pretraining on BERTimbau
using all mammography reports (train + test).  Run this BEFORE fine-tuning
exp020 to close the vocabulary/style gap between web-crawl pretraining and
Portuguese radiology text.

Reference: Gururangan et al. "Don't Stop Pretraining" (ACL 2020).

Usage:
    uv run python scripts/dapt_pretrain.py
    uv run python scripts/dapt_pretrain.py --epochs 5 --out models/bertimbau_dapt_5ep

The output directory (default: models/bertimbau_dapt/) is a standard
HuggingFace checkpoint.  Point any fine-tune config at it:
    model:
      pretrained: models/bertimbau_dapt
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="DAPT — continue BERTimbau MLM on mammography reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pretrained", default="neuralmind/bert-base-portuguese-cased",
                        help="Base checkpoint (HF ID or local path)")
    parser.add_argument("--out",        default="models/bertimbau_dapt",
                        help="Output directory for adapted checkpoint")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int,   default=32,
                        help="Per-device batch size")
    parser.add_argument("--max-length", type=int,   default=256)
    parser.add_argument("--mlm-prob",   type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    t0 = time.time()

    # ── Load texts ────────────────────────────────────────
    import pandas as pd
    from pathlib import Path

    from src.data import find_data_dir, load_train
    from src.features import clean_text

    data_dir = find_data_dir()
    train_df = load_train(data_dir)
    texts = train_df["report"].apply(clean_text).tolist()

    # Include test reports — unsupervised MLM, no labels needed
    test_path = Path(data_dir) / "test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_df.columns = test_df.columns.str.strip()
        texts += test_df["report"].apply(clean_text).tolist()
        print(f"Texts : {len(train_df):,} train + {len(test_df):,} test = {len(texts):,} total")
    else:
        print(f"Texts : {len(texts):,} (train only — test.csv not found locally)")

    # ── Load model + tokenizer ────────────────────────────
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from datasets import Dataset

    set_seed(args.seed)

    print(f"\nLoading : {args.pretrained}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    model     = AutoModelForMaskedLM.from_pretrained(args.pretrained)
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params  : {n_params:.0f}M")

    # ── Tokenise dataset ──────────────────────────────────
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(
        _tokenize,
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
        desc="Tokenising",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
    )

    # ── Training ──────────────────────────────────────────
    tmp_dir = os.path.join(args.out, "_tmp")
    os.makedirs(args.out, exist_ok=True)

    import torch
    fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=tmp_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=int(0.06 * (len(ds) // args.batch_size) * args.epochs),
        weight_decay=0.01,
        fp16=fp16,
        logging_steps=50,
        save_strategy="no",
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    print(f"\nDAPT config:")
    print(f"  epochs      : {args.epochs}")
    print(f"  lr          : {args.lr}")
    print(f"  batch/device: {args.batch_size}")
    print(f"  max_length  : {args.max_length}")
    print(f"  mlm_prob    : {args.mlm_prob}")
    print(f"  fp16        : {fp16}")
    print(f"  output      : {args.out}\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    trainer.train()

    # ── Save ──────────────────────────────────────────────
    # Save the full MLM model; downstream AutoModelForSequenceClassification.from_pretrained()
    # will drop the MLM head automatically (ignore_mismatched_sizes=True in build_model).
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    elapsed = time.time() - t0
    print(f"\nDAPT done  ({elapsed:.0f}s / {elapsed/60:.1f}m)")
    print(f"Checkpoint : {args.out}/")
    print(f"Next step  : uv run python run.py configs/exp020_bertimbau_dapt.yaml --train")


if __name__ == "__main__":
    main()
