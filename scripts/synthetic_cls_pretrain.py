#!/usr/bin/env python3
"""
Supervised synthetic classification pretraining: fine-tune a sequence
classifier on the competition-provided synthetic mammography reports, then
save the resulting checkpoint for downstream real-data fine-tuning.

Usage:
    uv run python scripts/synthetic_cls_pretrain.py
    uv run python scripts/synthetic_cls_pretrain.py --out models/bertimbau_synth_cls_5ep --epochs 5
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Supervised synthetic classification pretrain",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pretrained", default="neuralmind/bert-base-portuguese-cased",
                        help="Base checkpoint (HF ID or local path)")
    parser.add_argument("--out", default="models/bertimbau_synth_cls",
                        help="Output directory for the pretrained classification checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Per-device batch size")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", nargs="+", type=int, default=None,
                        help="Optional subset of target classes to pretrain on")
    args = parser.parse_args()

    t0 = time.time()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from src.data import load_synthetic
    from src.features import clean_text

    set_seed(args.seed)

    synth_df = load_synthetic(classes=args.classes)
    synth_df["text"] = synth_df["report"].apply(clean_text)

    print(f"Synthetic rows : {len(synth_df):,}")
    print(f"Class counts   :\n{synth_df['target'].value_counts().sort_index().to_string()}")
    print(f"Loading        : {args.pretrained}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained,
        num_labels=7,
        ignore_mismatched_sizes=True,
    )

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    ds = Dataset.from_pandas(synth_df[["text", "target"]].rename(columns={"target": "labels"}))
    ds = ds.map(
        _tokenize,
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
        desc="Tokenising",
    )

    tmp_dir = os.path.join(args.out, "_tmp")
    os.makedirs(args.out, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=tmp_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=max(1, int(0.06 * (len(ds) // args.batch_size) * args.epochs)),
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="no",
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("\nSynthetic supervised pretrain config:")
    print(f"  epochs      : {args.epochs}")
    print(f"  lr          : {args.lr}")
    print(f"  batch/device: {args.batch_size}")
    print(f"  max_length  : {args.max_length}")
    print(f"  output      : {args.out}\n")

    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    elapsed = time.time() - t0
    print(f"\nSynthetic pretrain done  ({elapsed:.0f}s / {elapsed/60:.1f}m)")
    print(f"Checkpoint : {args.out}/")


if __name__ == "__main__":
    main()
