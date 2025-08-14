# XNode SLM — Small Language Model for Agent Routing & Communication

This repo is an end-to-end reference for intent-driven **agent routing**:
- **Model server** (FastAPI + PyTorch/Transformers) classifies user intent and returns **top-k** confidences.
- **Backend** (Node/Express) applies thresholds and intent→agent mapping, and exposes consolidated **metrics**.
- **Frontend** (Angular) provides a simple UI to submit text and visualize predicted intent, score, **top-k**, and latency.
- **Dashboard** (Dash/Plotly) shows live charts for intents, latency, and fallback rate.

---

## Architecture

```
Angular UI (4200)  ->  Backend (3001)  ->  Model Server (8000)
                                   ^
                               Dashboard (8050) -> Backend /metrics
```

**Why this design?**
- The model server stays focused on inference.
- The backend owns routing policy (thresholds, mappings) and aggregates metrics.
- The UI remains thin and environment-configurable.
- The dashboard provides observability to tune routing behavior.

---

## Tech Stack

- **Model:** Python 3.12, PyTorch, Hugging Face Transformers/Datasets, FastAPI, Uvicorn
- **Backend:** Node.js 20, Express, Axios, CORS, dotenv
- **Frontend:** Angular (standalone components), HttpClient
- **Dashboard:** Dash 3+, Plotly

---

## Project Structure

```
slm-routing/
├─ model/
│  ├─ train.py                # train DistilBERT (or other HF model) on SNIPS
│  ├─ serve.py                # FastAPI app: /predict, /metrics, /labels
│  ├─ artifacts/
│  │  ├─ intent_classifier/   # saved model + tokenizer
│  │  ├─ labels.json          # label2id / id2label maps
│  │  └─ metrics.json         # eval metrics from training
│  └─ .venv/                  # python virtualenv (local)
├─ backend/
│  ├─ index.js                # Express server: /route, /metrics, /config
│  └─ .env                    # MODEL_URL, THRESHOLD, PORT
├─ frontend/angular/
│  ├─ src/
│  │  ├─ app/                 # App, template, styles
│  │  ├─ assets/env.js        # window.env.BACKEND_URL
│  │  └─ main.ts, index.html
│  ├─ package.json, angular.json, tsconfig*.json
├─ dashboard/
│  └─ app.py                  # Dash/Plotly live metrics dashboard
└─ README.md
```

---

## Quickstart

### Prerequisites
- Python **3.12**
- Node.js **20.x** + npm
- Angular CLI (`npm i -g @angular/cli`)
- macOS on Apple Silicon can use **MPS** acceleration automatically.

> If you used Anaconda previously, ensure `conda deactivate` before activating the project’s venv.

### 1) Train the model (once)

```bash
cd model
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers datasets scikit-learn fastapi uvicorn

# Train DistilBERT on SNIPS from the HF Hub
python train.py --model_name distilbert-base-uncased --epochs 3 --batch_size 16 --max_length 128
```

Artifacts are written to `model/artifacts/`:
- `artifacts/intent_classifier/` (model weights + tokenizer)
- `artifacts/labels.json` (id↔label mapping)
- `artifacts/metrics.json` (eval metrics)

> Alternatives: `prajjwal1/bert-tiny` (very fast) or `microsoft/deberta-v3-small` (higher accuracy; requires `pip install protobuf`).

### 2) Serve the model (FastAPI)

From the repo root:

```bash
# ensure package discovery
touch model/__init__.py

# activate the same venv used for training
source model/.venv/bin/activate

# run the model server
python -m uvicorn model.serve:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl -s http://127.0.0.1:8000/ | jq
curl -s -X POST 'http://127.0.0.1:8000/predict?top_k=3' \
  -H 'content-type: application/json' \
  -d '{"text":"book a table for two at 7pm"}' | jq
```

### 3) Start the backend (routing API)

```bash
cd backend
npm i
cat > .env <<'ENV'
MODEL_URL=http://127.0.0.1:8000
THRESHOLD=0.6
PORT=3001
ENV

node index.js
```

Smoke tests:

```bash
curl -s http://127.0.0.1:3001/healthz | jq
curl -s http://127.0.0.1:3001/config  | jq
curl -s -X POST http://127.0.0.1:3001/route \
  -H 'content-type: application/json' \
  -d '{"text":"play some jazz"}' | jq
```

### 4) Start the Angular UI

```bash
cd frontend/angular
npm i

# point UI to backend
mkdir -p src/assets
cat > src/assets/env.js <<'JS'
(window).env = { BACKEND_URL: 'http://127.0.0.1:3001' };
JS

ng serve
# open http://localhost:4200
```

Type a request, click **Route**, and view predicted **intent**, **score**, **top-k**, and **latency**.

### 5) Start the Dashboard

```bash
# reuse the model venv
source model/.venv/bin/activate
pip install dash plotly requests

cd dashboard
BACKEND_URL=http://127.0.0.1:3001 python app.py
# open http://127.0.0.1:8050
```

---

## API Contracts

### Model Server (FastAPI)

- `GET /` → `{ ok, service, labels }`
- `GET /labels` → `{ labels: string[] }`
- `GET /metrics` → eval metrics JSON from training
- `POST /predict?top_k=K`
  **Request**
  ```json
  { "text": "book a table for two at 7pm" }
  ```
  **Response**
  ```json
  {
    "intent": "BookRestaurant",
    "score": 0.9995,
    "top_k": [
      {"intent":"BookRestaurant","score":0.9995},
      {"intent":"SearchCreativeWork","score":0.0003},
      {"intent":"GetWeather","score":0.0002}
    ],
    "response": "BookRestaurant detected — connecting agent."
  }
  ```

### Backend (Express)

- `GET /healthz` → `{ ok, model, threshold }`
- `GET /config` → `{ model_url, threshold, labels }`
- `GET /metrics` → live routing metrics
  ```json
  {
    "service": "slm-backend",
    "uptime_s": 1234,
    "total_requests": 42,
    "fallback_rate": 0.0,
    "avg_latency_ms": 37.2,
    "p50_latency_ms": 34,
    "p95_latency_ms": 60,
    "intents": { "BookRestaurant": 20, "PlayMusic": 15, "GetWeather": 7 },
    "recent": [
      {
        "ts": "2025-08-13T19:01:02.123Z",
        "text": "book a table for two at 7pm",
        "intent": "BookRestaurant",
        "score": 0.9995,
        "ms": 33,
        "top_k": [ ... ]
      }
    ]
  }
  ```
- `POST /route`
  ```json
  { "text": "what's the weather in Boston tomorrow" }
  ```
  **Response**
  ```json
  {
    "intent": "GetWeather",
    "score": 0.992,
    "top_k": [...],
    "agent": { "name": "GetWeatherAgent", "endpoint": "/agents/generic" },
    "message": "GetWeather detected — connecting agent.",
    "latency_ms": 35
  }
  ```

---

## Sample Frontend Output

Example after submitting `“book a table for two at 7pm”`:

```
Intent: BookRestaurant (0.9995)
Latency: 33 ms
Top-k:
  • BookRestaurant — 0.9995
  • SearchCreativeWork — 0.0003
  • GetWeather — 0.0002
Message: BookRestaurant detected — connecting agent.
```

---

## What We Built & Why

- **Small language model** fine-tuned for **intent classification** on SNIPS.
- **Top-k** confidence to surface ambiguity and support safer routing decisions.
- **Routing policy** with a configurable **confidence threshold** and intent→agent mapping.
- **Observability** (metrics + dashboard) to guide threshold tuning and measure UX (latency, fallback rate).
- **Separation of concerns**: inference vs. policy vs. presentation.

This setup aligns with the project rubric:
- High accuracy on an intent dataset; efficient inference.
- Seamless integration into a workflow via the backend.
- Clear visualizations of metrics and a usable web UI.
- Clean, modular code with documented setup and usage.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'model'`**  
  Run Uvicorn from repo root with `python -m uvicorn model.serve:app ...` and ensure `model/__init__.py` exists.

- **DeBERTa tokenizer / protobuf error**  
  `pip install protobuf` or use `distilbert-base-uncased`.

- **CORS / 405**  
  Backend must use `app.use(cors())` and the UI must **POST** to `/route`.

- **Angular pipes (| number)**  
  Import `DecimalPipe` in the standalone component’s `imports` array.

- **Dash ≥ 3**  
  Use `app.run(...)`, not `run_server`.

---

## Environment & .gitignore

**Backend `.env`:**
```
MODEL_URL=http://127.0.0.1:8000
THRESHOLD=0.6
PORT=3001
```

**Frontend:**
- Commit `src/assets/env.sample.js`, and optionally gitignore the real `env.js` if it differs per environment.

**Recommended `.gitignore`:**
```
# Python
model/.venv/
__pycache__/

# Large weights (or use Git LFS)
model/artifacts/intent_classifier/pytorch_model.bin
model/artifacts/intent_classifier/model.safetensors

# Node/Angular
backend/node_modules/
frontend/angular/node_modules/
frontend/angular/.angular/
frontend/angular/dist/

# IDE
.vscode/
.DS_Store
```

---

## End-to-End Run

Open 4 terminals:

```bash
# 1) Model server
source model/.venv/bin/activate
python -m uvicorn model.serve:app --port 8000 --reload

# 2) Backend
cd backend && node index.js

# 3) Angular
cd frontend/angular && ng serve

# 4) Dashboard
source ../../model/.venv/bin/activate
cd ../../dashboard && BACKEND_URL=http://127.0.0.1:3001 python app.py
```

Send some requests to populate metrics:
```bash
curl -s -X POST http://127.0.0.1:3001/route -H 'content-type: application/json' -d '{"text":"book a table for two"}' | jq
curl -s -X POST http://127.0.0.1:3001/route -H 'content-type: application/json' -d '{"text":"play some jazz"}' | jq
```

Open:
- UI → http://localhost:4200
- Dashboard → http://127.0.0.1:8050
