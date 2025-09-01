# app.py
import os, json, subprocess, sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROOT = Path(__file__).parent
PY = sys.executable  # current venv python
PIPE = str(ROOT / "fx_pipeline.py")
DATA = str(ROOT / "data")
MODELS = str(ROOT / "models")
MT5_CFG = ROOT / "data" / "mt5_config.json"

Path(DATA).mkdir(parents=True, exist_ok=True)
Path(MODELS).mkdir(parents=True, exist_ok=True)

def run_pipe(*args):
    """Run fx_pipeline.py with args; return (ok, stdout, stderr)."""
    cmd = [PY, "-u", PIPE, *args]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    ok = (p.returncode == 0)
    return ok, (p.stdout or "").strip(), (p.stderr or "").strip()

@app.get("/api/health")
def health():
    return jsonify({"ok": True})

@app.post("/api/build")
def api_build():
    ticks_glob = request.json.get("ticks_glob", "data/ticks/*.csv")
    ok, out, err = run_pipe("build", "--ticks-glob", ticks_glob)
    return jsonify({"ok": ok, "stdout": out, "stderr": err})

@app.post("/api/train")
def api_train():
    ok, out, err = run_pipe("train")
    # fx_pipeline prints JSON metrics last; try to parse the last {...}
    metrics = None
    for line in reversed(out.splitlines()):
        if line.strip().startswith("{"):
            try:
                metrics = json.loads(line)
                break
            except Exception:
                pass
    return jsonify({"ok": ok, "metrics": metrics, "stdout": out, "stderr": err})

@app.get("/api/predict")
def api_predict():
    objective = request.args.get("objective", "acc")
    ok, out, err = run_pipe("predict", "--objective", objective)
    preds = None
    try:
        preds = json.loads(out.splitlines()[-1])
    except Exception:
        pass
    return jsonify({"ok": ok, "objective": objective, "predictions": preds, "stdout": out, "stderr": err})

@app.post("/api/holdout")
def api_holdout():
    cutoff = request.json.get("cutoff")
    last_days = request.json.get("last_days")
    args = ["holdout"]
    if cutoff:
        args += ["--cutoff", cutoff]
    elif last_days:
        args += ["--last-days", str(int(last_days))]
    else:
        return jsonify({"ok": False, "error": "Provide cutoff or last_days"}), 400
    ok, out, err = run_pipe(*args)
    res = None
    try:
        res = json.loads(out.splitlines()[-1])
    except Exception:
        pass
    return jsonify({"ok": ok, "result": res, "stdout": out, "stderr": err})

# ---- MetaTrader 5 helpers (host machine only) ----
try:
    import MetaTrader5 as mt5  # pip install MetaTrader5
except Exception:
    mt5 = None

@app.post("/api/mt5/config")
def save_mt5_config():
    """Save MT5 connection info locally (NOT checked into git)."""
    MT5_CFG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "login": request.json.get("login", ""),
        "password": request.json.get("password", ""),
        "server": request.json.get("server", ""),
        "path": request.json.get("path", ""),
        "symbol": request.json.get("symbol", "EURUSD"),
        "days": int(request.json.get("days", 1)),
    }
    MT5_CFG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return jsonify({"ok": True})

@app.post("/api/mt5/fetch_ticks")
def fetch_mt5_ticks():
    """Pull recent ticks from the local MT5 terminal and save a CSV under data/ticks/."""
    if mt5 is None:
        return jsonify({"ok": False, "error": "MetaTrader5 package not installed"}), 400

    cfg = {}
    if MT5_CFG.exists():
        try: cfg = json.loads(MT5_CFG.read_text(encoding="utf-8"))
        except Exception: cfg = {}
    cfg.update({k: request.json.get(k, cfg.get(k)) for k in
                ["login","password","server","path","symbol","days"]})
    login = int(cfg.get("login") or 0)
    password = cfg.get("password")
    server = cfg.get("server")
    path = cfg.get("path") or None
    symbol = cfg.get("symbol") or "EURUSD"
    days = int(cfg.get("days") or 1)

    # initialize/login
    if path:
        ok_init = mt5.initialize(path)
    else:
        ok_init = mt5.initialize()
    if not ok_init:
        return jsonify({"ok": False, "error": f"mt5.initialize failed: {mt5.last_error()}"}), 500
    if login and password and server:
        if not mt5.login(login=login, password=password, server=server):
            return jsonify({"ok": False, "error": f"mt5.login failed: {mt5.last_error()}"}), 500

    # timeframe: last N days of ticks
    from datetime import datetime, timedelta, timezone
    to_utc = datetime.now(timezone.utc)
    from_utc = to_utc - timedelta(days=days)
    ticks = mt5.copy_ticks_range(symbol, from_utc, to_utc, mt5.COPY_TICKS_ALL)
    mt5.shutdown()

    if ticks is None or len(ticks) == 0:
        return jsonify({"ok": False, "error": "no ticks returned"}), 404

    import pandas as pd
    df = pd.DataFrame(ticks)
    if "time_msc" in df.columns and df["time_msc"].notna().any():
        ts = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    else:
        ts = pd.to_datetime(df["time"], unit="s", utc=True)
    # prefer mid from bid/ask if present
    if {"bid","ask"}.issubset(df.columns):
        mid = (df["bid"] + df["ask"]) / 2.0
    else:
        # fallback to 'last' or 'price' if available
        mid = df.get("last", df.get("price", df.get("bid", df.get("ask", None)))))
        if mid is None:
            return jsonify({"ok": False, "error": "cannot infer mid price from MT5 ticks"}), 500

    out = Path(DATA) / "ticks"
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{symbol}_MT5_{to_utc.strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame({"time_utc": ts, "mid": mid}).to_csv(fname, index=False)
    return jsonify({"ok": True, "file": str(fname)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
