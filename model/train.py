#!/usr/bin/env python3
# Trains a small transformer on SNIPS pulled from Hugging Face (no local files).
import os, json, argparse, pandas as pd, numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTI = os.path.join(ROOT, "model", "artifacts")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="prajjwal1/bert-tiny")  # fast smoke test; swap to distilbert-base-uncased later
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--hf_dataset", default="benayas/snips")     # has columns: text, category
    ap.add_argument("--hf_text_col", default="text")
    ap.add_argument("--hf_label_col", default="category")
    args = ap.parse_args()

    os.makedirs(ARTI, exist_ok=True)

    ds = load_dataset(args.hf_dataset)
    df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)
    df = df[[args.hf_text_col, args.hf_label_col]].rename(columns={args.hf_text_col:"text", args.hf_label_col:"intent"})

    label_list = sorted(df["intent"].unique())
    label2id = {l:i for i,l in enumerate(label_list)}
    id2label = {i:l for l,i in label2id.items()}
    df["label"] = df["intent"].map(label2id)

    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    tok = AutoTokenizer.from_pretrained(args.model_name)
    def encode_one(txt):
        return tok(txt, truncation=True, padding="max_length", max_length=64, return_tensors="pt")

    import torch
    class TorchDS(torch.utils.data.Dataset):
        def __init__(self, pdf): self.pdf = pdf.reset_index(drop=True)
        def __len__(self): return len(self.pdf)
        def __getitem__(self, i):
            bt = encode_one(self.pdf.loc[i, "text"])
            item = {k:v.squeeze(0) for k,v in bt.items()}
            item["labels"] = torch.tensor(int(self.pdf.loc[i, "label"]))
            return item

    train_ds, test_ds = TorchDS(tr), TorchDS(te)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    targs = TrainingArguments(
        output_dir=os.path.join(ARTI, "runs"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=test_ds,
                      compute_metrics=compute_metrics, tokenizer=tok)
    trainer.train()
    metrics = trainer.evaluate()

    os.makedirs(os.path.join(ARTI, "intent_classifier"), exist_ok=True)
    model.save_pretrained(os.path.join(ARTI, "intent_classifier"))
    tok.save_pretrained(os.path.join(ARTI, "intent_classifier"))
    with open(os.path.join(ARTI, "labels.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    with open(os.path.join(ARTI, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("DONE metrics:", metrics)

if __name__ == "__main__":
    main()
