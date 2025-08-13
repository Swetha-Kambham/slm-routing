# model/serve.py
import os, json, pathlib
from typing import Optional, List

import torch
from torch.nn.functional import softmax
from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTI_DIR = ROOT / "model" / "artifacts"
OUT_DIR  = ARTI_DIR / "intent_classifier"
MAX_LEN  = int(os.getenv("MAX_LEN", "128"))

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = pick_device()

# --- load labels/metrics (if present) ---
labels_path  = ARTI_DIR / "labels.json"
metrics_path = ARTI_DIR / "metrics.json"
id2label, label2id = {}, {}
if labels_path.exists():
    with open(labels_path, "r") as f:
        m = json.load(f)
        label2id = m.get("label2id", {})
        # keys might be str or int; normalize
        id2label = {str(v): k for k, v in label2id.items()}
        if not id2label:
            id2label = m.get("id2label", {})
            label2id = {v: int(k) for k, v in id2label.items()}

# optional canned responses
RESPONSES = {}
resp_path = ROOT / "data" / "responses.json"
if resp_path.exists():
    try:
        with open(resp_path) as f:
            RESPONSES = json.load(f)
    except Exception:
        RESPONSES = {}

# --- load model/tokenizer ---
tok = AutoTokenizer.from_pretrained(str(OUT_DIR), use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(str(OUT_DIR))
model.eval().to(DEVICE)

app = FastAPI(title="SLM Intent Model", version="1.0.0")

class PredictIn(BaseModel):
    text: str

@app.get("/")
def root():
    return {"ok": True, "service": "SLM Intent Model", "labels": [id2label[str(i)] for i in range(len(id2label))]}

@app.get("/labels")
def labels():
    return {"labels": [id2label[str(i)] for i in range(len(id2label))]}

@app.get("/metrics")
def metrics():
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {"status": "no-metrics"}

@app.post("/predict")
def predict(req: PredictIn, top_k: int = Query(1, ge=1, le=10)):
    text = (req.text or "").strip()
    if not text:
        return {"error": "empty_text"}

    enc = tok(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        out = model(**enc)
        logits = out.logits.squeeze(0)  # [num_labels]
        probs  = softmax(logits, dim=-1).detach().cpu().tolist()

    # rank by probability
    pairs = [(id2label[str(i)], float(p)) for i, p in enumerate(probs)]
    pairs.sort(key=lambda x: x[1], reverse=True)

    best_label, best_score = pairs[0]
    msg_list = RESPONSES.get(best_label) or [f"{best_label} detected — connecting agent."]
    msg = msg_list[0] if isinstance(msg_list, list) else str(msg_list)

    return {
        "intent": best_label,
        "score": best_score,
        "top_k": [{"intent": lab, "score": sc} for lab, sc in pairs[:top_k]],
        "response": msg,
    }
