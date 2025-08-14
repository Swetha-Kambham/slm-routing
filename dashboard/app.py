import os
import time
import threading
import requests
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

#Config
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:3001")
METRICS_URL = f"{BACKEND}/metrics"
CONFIG_URL  = f"{BACKEND}/config"
POLL_SEC    = float(os.getenv("POLL_SEC", "2"))

# Simple cache updated by a background thread
cache = {"metrics": {}, "config": {}}

def poll():
    while True:
        try:
            cache["metrics"] = requests.get(METRICS_URL, timeout=2).json()
        except Exception:
            cache["metrics"] = {}
        try:
            cache["config"]  = requests.get(CONFIG_URL, timeout=2).json()
        except Exception:
            cache["config"]  = {}
        time.sleep(POLL_SEC)

threading.Thread(target=poll, daemon=True).start()

# ====== Dash app ======
app = Dash(__name__)
app.title = "SLM Routing Dashboard"

app.layout = html.Div(
    [
        html.H2("SLM Routing Dashboard"),

        html.Div(
            id="config",
            style={"padding":"8px","border":"1px solid #ddd","borderRadius":"8px","marginBottom":"12px"}
        ),

        # KPI cards (numbers + gauge) in a 2x3 grid (no rowspan/columnspan)
        dcc.Graph(id="kpi-cards", style={"height":"360px", "marginBottom":"16px"}),

        dcc.Graph(id="intent-bar", style={"height":"380px"}),

        html.H3("Recent routed events"),
        html.Div(id="recent-table"),

        dcc.Interval(id="tick", interval=int(POLL_SEC * 1000), n_intervals=0),
    ],
    style={"maxWidth":"1100px","margin":"24px auto","fontFamily":"system-ui, Arial"}
)

#Callbacks
@app.callback(Output("config","children"), Input("tick","n_intervals"))
def show_cfg(_):
    c = cache.get("config") or {}
    model_url = c.get("model_url", "?")
    threshold = c.get("threshold", "?")
    labels = ", ".join(c.get("labels", [])) if c.get("labels") else "(unknown)"
    return html.Div([
        html.Div(f"Backend URL: {BACKEND}"),
        html.Div(f"Model URL: {model_url}"),
        html.Div(f"Threshold: {threshold}"),
        html.Div(f"Labels: {labels}"),
    ])

@app.callback(Output("kpi-cards","figure"), Input("tick","n_intervals"))
def kpis(_):
    m = cache.get("metrics") or {}
    total = int(m.get("total_requests", 0) or 0)
    fb_rate = float(m.get("fallback_rate", 0) or 0) * 100.0
    avg = float(m.get("avg_latency_ms", 0) or 0)
    p50 = float(m.get("p50_latency_ms", 0) or 0)
    p95 = float(m.get("p95_latency_ms", 0) or 0)

    fig = go.Figure()
    # Row 0
    fig.add_trace(go.Indicator(
        mode="number", value=total, title={"text":"Total requests"},
        domain={"row": 0, "column": 0}
    ))
    fig.add_trace(go.Indicator(
        mode="number", value=avg, title={"text":"Avg latency (ms)"},
        domain={"row": 0, "column": 1}
    ))
    fig.add_trace(go.Indicator(
        mode="number", value=p50, title={"text":"p50 latency (ms)"},
        domain={"row": 0, "column": 2}
    ))
    # Row 1
    fig.add_trace(go.Indicator(
        mode="number", value=p95, title={"text":"p95 latency (ms)"},
        domain={"row": 1, "column": 0}
    ))
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=fb_rate,
        title={"text":"Fallback rate (%)"},
        gauge={"axis": {"range": [0, 100]}},
        domain={"row": 1, "column": 1}
    ))


    fig.update_layout(
        grid={"rows": 2, "columns": 3, "pattern": "independent"},
        height=360, margin=dict(l=20, r=20, t=20, b=10)
    )
    return fig

@app.callback(Output("intent-bar","figure"), Input("tick","n_intervals"))
def intent_hist(_):
    m = cache.get("metrics") or {}
    hist = m.get("intents", {}) or {}
    intents = list(hist.keys())
    counts = [hist[k] for k in intents]
    fig = go.Figure(go.Bar(x=intents, y=counts))
    fig.update_layout(
        title="Intent count (since server start)",
        xaxis_title="Intent",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=40, b=40)
    )
    return fig

@app.callback(Output("recent-table","children"), Input("tick","n_intervals"))
def recent_tbl(_):
    m = cache.get("metrics") or {}
    rows = m.get("recent", []) or []
    if not rows:
        return html.Div(
            "No recent events yet. Try sending a few /route requests from the UI.",
            style={"color":"#666"}
        )

    # Build a very simple HTML table (last 20 events)
    head = html.Thead(
        html.Tr([html.Th("Time"), html.Th("Text"), html.Th("Intent"), html.Th("Score"), html.Th("Latency ms")])
    )
    body = html.Tbody([
        html.Tr([
            html.Td(r.get("ts","")[:19].replace("T"," ")),
            html.Td(r.get("text","")),
            html.Td(r.get("intent","")),
            html.Td(f"{float(r.get('score',0)):.3f}"),
            html.Td(str(r.get("ms",""))),
        ]) for r in reversed(rows[-20:])
    ])

    return html.Table(
        [head, body],
        style={
            "width":"100%",
            "borderCollapse":"collapse",
            "border":"1px solid #eee"
        }
    )

if __name__ == "__main__":
    # Dash >= 3: use app.run(...)
    app.run(debug=True, host="127.0.0.1", port=8050)
