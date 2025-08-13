#!/usr/bin/env python3
# FastAPI server for the trained intent classifier
import os, json, random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTI = os.path.join(ROOT, "model", "artifacts")
MODEL_DIR = os.path.join(ARTI, "intent_classifier")
LABELS_PATH = os.path.join(ARTI, "labels.json")
METRICS_PATH = os.path.join(ARTI, "metrics.json")
RESPONSES_PATH = os.path.join(ROOT, "data", "responses.json")

if not os.path.isdir(MODEL_DIR):
    raise RuntimeError(f"Model dir not found: {MODEL_DIR}. Did you run train.py?")

with open(LABELS_PATH) as f:
    labs = json.load(f)
id2label = {int(k): v for k, v in labs["id2label"].items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
model.to(device)
model.eval()

# Intent → canned responses
if os.path.exists(RESPONSES_PATH):
    RESPONSES = json.load(open(RESPONSES_PATH))
else:
    RESPONSES = {v: [f"Routing to agent for {v}.", f"{v} detected — connecting agent."] for v in id2label.values()}

class Inp(BaseModel):
    text: str

app = FastAPI(title="SLM Intent Server")

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "SLM Intent Server",
        "endpoints": {"POST /predict": {"body": {"text": "string"}}, "GET /metrics": {}}
    }

@app.post("/predict")
def predict(inp: Inp):
    txt = (inp.text or "").strip()
    if not txt:
        raise HTTPException(status_code=400, detail="empty text")
    enc = tokenizer(txt, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0)
        score, idx = torch.max(probs, dim=-1)
    intent = id2label[idx.item()]
    msg = random.choice(RESPONSES.get(intent, ["Routing to the best agent."]))
    return {"intent": intent, "score": float(score.item()), "response": msg}

@app.get("/metrics")
def metrics():
    return json.load(open(METRICS_PATH)) if os.path.exists(METRICS_PATH) else {"status": "no-metrics"}
