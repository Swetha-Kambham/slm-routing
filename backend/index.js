// backend/index.js
// Simple routing API that calls the model server and decides which agent to use.

const express = require("express");
const axios = require("axios");
const cors = require("cors");
const morgan = require("morgan");
require('dotenv').config();

const app = express();
app.use(cors());            // allow UI calls from localhost:4200
app.use(express.json());    // parse JSON bodies
app.use(morgan("dev"));     // pretty logs

// where your FastAPI model is running:
const MODEL_URL = process.env.MODEL_URL || "http://127.0.0.1:8000";
// route to fallback when confidence is low:
const THRESHOLD = Number(process.env.THRESHOLD || 0.6);

// Optional static mappings: intent -> agent metadata
// Add entries here if you want custom agent names/endpoints
const AGENTS = {
  // "BookRestaurant": { name: "ReservationsAgent", endpoint: "/agents/reservations" },
  // "GetWeather": { name: "WeatherAgent", endpoint: "/agents/weather" },
};

function resolveAgent(intent, score) {
  // if confidence too low, send to fallback
  if (Number.isFinite(score) && score < THRESHOLD) {
    return { name: "FallbackAgent", endpoint: "/agents/generic" };
  }
  // use static map if provided; else generic "<Intent>Agent"
  return AGENTS[intent] || { name: `${intent}Agent`, endpoint: "/agents/generic" };
}

// Health check
app.get("/healthz", (req, res) => res.json({ ok: true, model: MODEL_URL, threshold: THRESHOLD }));

// Main routing endpoint
app.post("/route", async (req, res) => {
  try {
    const { text } = req.body || {};
    if (!text || !text.trim()) {
      return res.status(400).json({ error: "missing text" });
    }
    // ask the model for intent + score
    const { data } = await axios.post(`${MODEL_URL}/predict?top_k=3`, { text });
    const { intent, score, response, top_k } = data;
    const agent = resolveAgent(intent, score);
    return res.json({ intent, score, top_k, agent, message: response });
  } catch (err) {
    console.error("route error:", err.message || err);
    return res.status(500).json({ error: "routing_failed" });
  }
});

// Optional: pretend to call an agent (simulated latency)
app.post("/agents/:name", async (req, res) => {
  const ms = 200 + Math.floor(Math.random() * 600);
  await new Promise(r => setTimeout(r, ms));
  res.json({ agent: req.params.name, processedInMs: ms, status: "ok" });
});

// Proxy model metrics for the dashboard
app.get("/metrics", async (_req, res) => {
  try {
    const { data } = await axios.get(`${MODEL_URL}/metrics`, { timeout: 1500 });
    res.json(data);
  } catch (e) {
    res.json({ status: "no-metrics" });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Backend listening on http://127.0.0.1:${PORT}`);
  console.log(`Using MODEL_URL=${MODEL_URL}  THRESHOLD=${THRESHOLD}`);
});
