#!/usr/bin/env python3
import os, json, random
from fastapi import FastAPI
from pydantic import BaseModel

INTENTS = ["AddToPlaylist","BookRestaurant","GetWeather","PlayMusic","RateBook","SearchCreativeWork","SearchScreeningEvent"]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESP_PATH = os.path.join(ROOT, "data", "responses.json")

if os.path.exists(RESP_PATH):
    RESPONSES = json.load(open(RESP_PATH))
else:
    RESPONSES = {i: [f"Routing to agent for {i}.", f"{i} detected — connecting agent."] for i in INTENTS}

class Inp(BaseModel):
    text: str

app = FastAPI(title="SLM Mock Server")

@app.post("/predict")
def predict(inp: Inp):
    intent = random.choice(list(RESPONSES.keys()))
    score = round(random.uniform(0.55, 0.98), 4)
    msg = random.choice(RESPONSES[intent])
    return {"intent": intent, "score": score, "response": msg}

@app.get("/metrics")
def metrics():
    return {"eval_accuracy": 0.90, "eval_precision": 0.90, "eval_recall": 0.90, "eval_f1": 0.90}
