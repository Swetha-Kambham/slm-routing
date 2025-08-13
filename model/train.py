#!/usr/bin/env python3
"""
Train a small transformer on SNIPS directly from Hugging Face (no local dataset).
Robust to older `transformers` by falling back when some TrainingArguments fields
(e.g., evaluation_strategy) are unsupported.

Usage examples:
  python train.py --model_name prajjwal1/bert-tiny
  python train.py --model_name distilbert-base-uncased
  python train.py --model_name microsoft/deberta-v3-small --epochs 3 --batch_size 16 --max_length 128
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd

# HF datasets
from datasets import load_dataset

# metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# HF transformers
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# torch runtime
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTI_DIR = os.path.join(ROOT, "model", "artifacts")
OUT_DIR = os.path.join(ARTI_DIR, "intent_classifier")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def pick_device():
    # auto device: CUDA → MPS (Apple) → CPU
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name",
        default="microsoft/deberta-v3-small",
        help="Model name (e.g., prajjwal1/bert-tiny | distilbert-base-uncased | microsoft/deberta-v3-small)",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=128)

    # dataset config (from Hugging Face)
    ap.add_argument("--hf_dataset", default="benayas/snips", help="HF dataset id")
    ap.add_argument("--hf_text_col", default="text")
    ap.add_argument("--hf_label_col", default="category")

    # optional: quick subsampling (useful for smoke tests)
    ap.add_argument("--limit_train", type=int, default=0, help="use only N train examples")
    ap.add_argument("--limit_eval", type=int, default=0, help="use only N eval examples")

    args = ap.parse_args()

    os.makedirs(ARTI_DIR, exist_ok=True)

    # Debug info to ensure the right env/version is running
    print(f"DEBUG transformers: {transformers.__version__}  python: {sys.executable}")
    print(f"DEBUG device: {pick_device()}  model: {args.model_name}")

    # 1) Load dataset from the Hub
    ds = load_dataset(args.hf_dataset)  # expects splits with columns like text + category
    df = pd.concat(
        [ds["train"].to_pandas(), ds["test"].to_pandas()],
        ignore_index=True,
    )
    df = df[[args.hf_text_col, args.hf_label_col]].rename(
        columns={args.hf_text_col: "text", args.hf_label_col: "intent"}
    )

    # 2) Labels ↔ ids
    label_list = sorted(df["intent"].unique())
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    df["label"] = df["intent"].map(label2id)

    # 3) Split
    tr, te = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    if args.limit_train and args.limit_train < len(tr):
        tr = tr.sample(n=args.limit_train, random_state=42)
    if args.limit_eval and args.limit_eval < len(te):
        te = te.sample(n=args.limit_eval, random_state=42)

    # 4) Tokenizer (slow path is safest across environments)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    def encode_one(txt: str):
        return tok(
            txt,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )

    class TorchDS(torch.utils.data.Dataset):
        def __init__(self, pdf: pd.DataFrame):
            self.pdf = pdf.reset_index(drop=True)

        def __len__(self):
            return len(self.pdf)

        def __getitem__(self, i: int):
            bt = encode_one(self.pdf.loc[i, "text"])
            item = {k: v.squeeze(0) for k, v in bt.items()}
            item["labels"] = torch.tensor(int(self.pdf.loc[i, "label"]))
            return item

    train_ds, eval_ds = TorchDS(tr), TorchDS(te)

    # 5) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # 6) Trainer args — robust to older transformers
    base_kwargs = dict(
        output_dir=os.path.join(ARTI_DIR, "runs"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=10,
        report_to=[],  # disable wandb/comet unless you add them on purpose
    )

    try:
        # Newer transformers (supports these args)
        targs = TrainingArguments(
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            **base_kwargs,
        )
    except TypeError:
        # Older transformers fallback (no eval/save strategies)
        targs = TrainingArguments(**base_kwargs)

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        tokenizer=tok,
    )

    # 8) Train + evaluate
    trainer.train()
    metrics = trainer.evaluate()
    print("FINAL metrics:", metrics)

    # 9) Save artifacts
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    with open(os.path.join(ARTI_DIR, "labels.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    with open(os.path.join(ARTI_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model + tokenizer → {OUT_DIR}")
    print(f"Saved label maps → {os.path.join(ARTI_DIR, 'labels.json')}")
    print(f"Saved metrics → {os.path.join(ARTI_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()
