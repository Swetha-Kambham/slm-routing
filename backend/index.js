// backend/index.js
// Routing API: calls the model server, applies thresholds/agent mapping, and exposes metrics.

const express = require("express");
const axios = require("axios");
const cors = require("cors");
const morgan = require("morgan");
const os = require("os");
require("dotenv").config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(morgan("dev"));

// Env/config
const MODEL_URL = process.env.MODEL_URL || "http://127.0.0.1:8000";
const THRESHOLD = Number(process.env.THRESHOLD || 0.6);
const PORT = process.env.PORT || 3001;

// Optional static intent → agent mapping
const AGENTS = {
  // "BookRestaurant": { name: "ReservationsAgent", endpoint: "/agents/reservations" },
  // "GetWeather":     { name: "WeatherAgent",     endpoint: "/agents/weather" },
};

function resolveAgent(intent, score) {
  if (Number.isFinite(score) && score < THRESHOLD) {
    return { name: "FallbackAgent", endpoint: "/agents/generic" };
  }
  return AGENTS[intent] || { name: `${intent}Agent`, endpoint: "/agents/generic" };
}

// ----------------- Metrics (for dashboard) -----------------
let startedAt = Date.now();
let total = 0;
let fallback = 0;
const latencies = [];               // keep last N latencies
const intents = Object.create(null); // histogram of predicted intents
const recent = [];                  // ring buffer of last N routed events
const MAX_LAT = 500;
const MAX_RECENT = 100;

const pushRing = (arr, item, cap) => { arr.push(item); if (arr.length > cap) arr.shift(); };
const avg = (arr) => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0);
const pct = (arr, p) => {
  if (!arr.length) return 0;
  const a = [...arr].sort((x, y) => x - y);
  const i = Math.max(0, Math.min(Math.ceil((p / 100) * a.length) - 1, a.length - 1));
  return a[i];
};
// -----------------------------------------------------------

// Health
app.get("/healthz", (_req, res) => res.json({ ok: true, model: MODEL_URL, threshold: THRESHOLD }));

// Expose model labels + threshold (for UI)
app.get("/config", async (_req, res) => {
  try {
    const { data } = await axios.get(`${MODEL_URL}/labels`, { timeout: 2000 });
    res.json({ model_url: MODEL_URL, threshold: THRESHOLD, labels: data.labels || [] });
  } catch {
    res.json({ model_url: MODEL_URL, threshold: THRESHOLD, labels: [] });
  }
});

// Main routing endpoint
app.post("/route", async (req, res) => {
  const t0 = Date.now();
  try {
    const { text } = req.body || {};
    if (!text || !text.trim()) {
      return res.status(400).json({ error: "missing text" });
    }

    // Ask the model for prediction (request top-3 for transparency)
    const { data } = await axios.post(`${MODEL_URL}/predict?top_k=3`, { text }, { timeout: 8000 });
    const { intent, score, response, top_k } = data;

    const agent = resolveAgent(intent, score);

    // ---- metrics update ----
    const ms = Date.now() - t0;
    total += 1;
    intents[intent] = (intents[intent] || 0) + 1;
    pushRing(latencies, ms, MAX_LAT);
    if (Number.isFinite(score) && score < THRESHOLD) fallback += 1;
    pushRing(
      recent,
      { ts: new Date().toISOString(), text: String(text).slice(0, 160), intent, score, top_k, agent, ms },
      MAX_RECENT
    );
    // ------------------------

    return res.json({ intent, score, top_k, agent, message: response, latency_ms: ms });
  } catch (err) {
    console.error("route error:", err?.message || err);
    return res.status(500).json({ error: "routing_failed" });
  }
});

// Simulated agent endpoint (optional)
app.post("/agents/:name", async (req, res) => {
  const ms = 200 + Math.floor(Math.random() * 600);
  await new Promise((r) => setTimeout(r, ms));
  res.json({ agent: req.params.name, processedInMs: ms, status: "ok" });
});

// Rich metrics for the dashboard
app.get("/metrics", (_req, res) => {
  res.json({
    service: "slm-backend",
    host: os.hostname(),
    uptime_s: Math.round((Date.now() - startedAt) / 1000),
    total_requests: total,
    fallback_rate: total ? +(fallback / total).toFixed(4) : 0,
    avg_latency_ms: +avg(latencies).toFixed(2),
    p50_latency_ms: pct(latencies, 50),
    p95_latency_ms: pct(latencies, 95),
    intents,   // { intent: count }
    recent,    // last N routed events (for debugging or dashboard table)
  });
});

app.listen(PORT, () => {
  console.log(`Backend listening on http://127.0.0.1:${PORT}`);
  console.log(`Using MODEL_URL=${MODEL_URL}  THRESHOLD=${THRESHOLD}`);
});
