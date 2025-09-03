# app.py
import os, sys, json, subprocess
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- env / paths ------------------------------------------------------------
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

ROOT   = Path(__file__).parent.resolve()
PY     = sys.executable                          # current venv python
PIPE   = str(ROOT / "fx_pipeline.py")
DATA   = ROOT / "data"
MODELS = ROOT / "models"
MT5_CFG = DATA / "mt5_config.json"

DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# Optional MT5 subprocess config (for .venv-mt5 usage)
MT5_PYTHON = os.getenv("MT5_PYTHON")  # e.g. C:\...\forex-app\.venv-mt5\Scripts\python.exe
MT5_SCRIPT = os.getenv("MT5_SCRIPT")  # e.g. C:\...\forex-app\mt5_collector.py
MT5_USE_SUBPROCESS = os.getenv("MT5_USE_SUBPROCESS", "").lower() in {"1", "true", "yes"}

# --- MetaTrader5 availability (module-scope) ---
try:
    import MetaTrader5 as mt5  # pip install MetaTrader5 (Windows only)
except Exception:
    mt5 = None


SERVE_UI = os.getenv("SERVE_UI", "1") == "1"
if SERVE_UI:
    app = Flask(__name__, static_folder="ui/dist", static_url_path="/")
else:
    app = Flask(__name__)

CORS(app)

# Serve index.html when hitting /
if SERVE_UI:
    @app.get("/")
    def _index():
        return app.send_static_file("index.html")


# --- app setup --------------------------------------------------------------
app = Flask(__name__)
CORS(app)


# --- helpers ----------------------------------------------------------------
def run_pipe(*args):
    """Run fx_pipeline.py with args; return (ok, stdout, stderr)."""
    cmd = [PY, "-u", PIPE, *args]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return p.returncode == 0, (p.stdout or "").strip(), (p.stderr or "").strip()

# --- helpers ---------------------------------------------------------------

def _parse_last_json_blob(s: str):
    """
    Try to parse JSON from multi-line stdout.
    1) direct json.loads
    2) substring between first '{' and last '}'
    3) accumulate lines from the end until it parses
    """
    if not s:
        return None
    s = s.strip()
    # 1) direct
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) wide slice
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = s[start:end+1]
        try:
            return json.loads(blob)
        except Exception:
            pass
    # 3) build from the bottom up
    lines = s.splitlines()
    buf = []
    for i in range(len(lines) - 1, -1, -1):
        buf.insert(0, lines[i])
        try:
            return json.loads("\n".join(buf))
        except Exception:
            continue
    return None


def _extract_json_blocks(text: str):
    """
    Best-effort: collect JSON objects printed in stdout by scanning for
    blocks that start with '{' on a line and end with '}' on a later line.
    """
    objs, buf, inside = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if not inside and s.startswith("{"):
            inside = True
            buf = [line]
        elif inside:
            buf.append(line)
            if s.endswith("}"):
                try:
                    obj = json.loads("\n".join(buf))
                    objs.append(obj)
                except Exception:
                    pass
                inside = False
                buf = []
    return objs

def _pick_metrics_and_policies(json_list):
    """
    Try to distinguish metrics vs. policies:
      - metrics have keys like 'AUC_mean' in nested window dicts
      - policies have keys like 'auc', 'flip', 'acc' in nested window dicts
    Returns (metrics, policies) where either may be None.
    """
    metrics = policies = None
    for obj in json_list:
        if not isinstance(obj, dict):
            continue
        # look into a known window key if present
        win = None
        for k in ("daily", "w2_10", "w3_12", "2_10"):
            if k in obj and isinstance(obj[k], dict):
                win = obj[k]
                break
        if not win:
            continue
        # heuristics
        if any(k.endswith("_mean") for k in win.keys()):
            metrics = obj
        elif {"auc", "flip", "acc"}.issubset(set(win.keys())):
            policies = obj
    # fallback: if only one object, treat it as metrics
    if not metrics and json_list:
        metrics = json_list[0]
    return metrics, policies

# --- helper: parse last JSON-looking line from a process' stdout ---
def _parse_last_json_line(text: str):
    """Return the last line of stdout that parses as JSON, else None."""
    for line in reversed((text or '').splitlines()):
        s = line.strip()
        if not s or s[0] not in '{[':
            continue
        try:
            return json.loads(s)
        except Exception:
            continue
    return None


# --- routes -----------------------------------------------------------------
@app.get("/api/health")
def health():
    return jsonify({"ok": True})

@app.post("/api/build")
def api_build():
    # accept either 'ticks_glob' or 'glob' from client
    body = request.get_json(silent=True) or {}
    ticks_glob = body.get("ticks_glob") or body.get("glob") or "data/ticks/*.csv"
    ok, out, err = run_pipe("build", "--ticks-glob", ticks_glob)
    return jsonify({"ok": ok, "stdout": out, "stderr": err})


@app.post("/api/train")
def api_train():
    ok, out, err = run_pipe("train")
    metrics = _parse_last_json_line(out)
    return jsonify({"ok": ok, "metrics": metrics, "stdout": out, "stderr": err})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    objective = request.args.get("objective", "acc")
    return jsonify(_do_predict(objective))

def _do_predict(objective: str):
    ok, out, err = run_pipe("predict", "--objective", objective)
    preds = _parse_last_json_blob(out)  # <-- use the robust parser here
    return {
        "ok": ok,
        "objective": objective,
        "predictions": preds,
        "stdout": out,
        "stderr": err,
    }

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
    res = _parse_last_json_line(out)
    return jsonify({"ok": ok, "result": res, "stdout": out, "stderr": err})

# ---- Docker control: single regular app service ----
APP_SVC = os.getenv("DOCKER_SVC_APP", "fx-app")

def run_compose(*args):
    """Run `docker compose <args>` in repo root."""
    try:
        p = subprocess.run(
            ["docker", "compose", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        ok = (p.returncode == 0)
        return ok, (p.stdout or "").strip(), (p.stderr or "").strip()
    except FileNotFoundError:
        return False, "", "docker not found (is Docker Desktop installed and on PATH?)"

@app.post("/api/docker/app/<string:op>")
def docker_app(op):
    """POST /api/docker/app/{start|stop|status}"""
    op = op.lower()
    if op in ("start", "up"):
        ok, out, err = run_compose("up", "-d", APP_SVC)
    elif op in ("stop", "down"):
        ok1, out1, err1 = run_compose("stop", APP_SVC)
        ok2, out2, err2 = run_compose("rm", "-f", APP_SVC)
        ok = ok1 and ok2
        out = "\n".join([out1, out2]).strip()
        err = "\n".join([err1, err2]).strip()
    elif op == "status":
        ok, out, err = run_compose("ps", APP_SVC)
    else:
        return jsonify({"ok": False, "error": f"unsupported op '{op}'"}), 400
    return jsonify({"ok": ok, "service": APP_SVC, "op": op, "stdout": out, "stderr": err})


# ---- MetaTrader 5: config + fetch ------------------------------------------
@app.post("/api/mt5/config")
def save_mt5_config():
    """Save MT5 connection info locally (NOT checked into git)."""
    MT5_CFG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "login":    request.json.get("login", ""),
        "password": request.json.get("password", ""),
        "server":   request.json.get("server", ""),
        "path":     request.json.get("path", ""),
        "symbol":   request.json.get("symbol", "EURUSD"),
        "days":     int(request.json.get("days", 1)),
    }
    MT5_CFG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "saved": str(MT5_CFG)})

def _read_mt5_cfg_and_merge(req_json: dict):
    cfg = {}
    if MT5_CFG.exists():
        try:
            cfg = json.loads(MT5_CFG.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    # overlay request values
    for k in ["login", "password", "server", "path", "symbol", "days"]:
        v = req_json.get(k)
        if v is not None:
            cfg[k] = v
    # defaults
    cfg.setdefault("symbol", "EURUSD")
    cfg["days"] = int(cfg.get("days") or 1)
    if cfg.get("login"):
        try:
            cfg["login"] = int(cfg["login"])
        except Exception:
            pass
    return cfg

def _mt5_fetch_via_package(cfg: dict):
    """Fetch using MetaTrader5 package in this venv."""
    if mt5 is None:
        return False, "MetaTrader5 package not installed in current venv", None

    login   = int(cfg.get("login") or 0)
    password= cfg.get("password")
    server  = cfg.get("server")
    path    = cfg.get("path") or None
    symbol  = cfg.get("symbol") or "EURUSD"
    days    = int(cfg.get("days") or 1)

    if path:
        ok_init = mt5.initialize(path)
    else:
        ok_init = mt5.initialize()
    if not ok_init:
        return False, f"mt5.initialize failed: {mt5.last_error()}", None

    if login and password and server:
        if not mt5.login(login=login, password=password, server=server):
            err = f"mt5.login failed: {mt5.last_error()}"
            mt5.shutdown()
            return False, err, None

    from datetime import datetime, timedelta, timezone
    to_utc = datetime.now(timezone.utc)
    from_utc = to_utc - timedelta(days=days)
    ticks = mt5.copy_ticks_range(symbol, from_utc, to_utc, mt5.COPY_TICKS_ALL)
    mt5.shutdown()

    if ticks is None or len(ticks) == 0:
        return False, "no ticks returned", None

    # Save CSV in data/ticks/
    import pandas as pd
    df = pd.DataFrame(ticks)
    if "time_msc" in df.columns and df["time_msc"].notna().any():
        ts = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    else:
        ts = pd.to_datetime(df["time"], unit="s", utc=True)

    # prefer mid from bid/ask if present; else fallbacks
    if {"bid", "ask"}.issubset(df.columns):
        mid = (df["bid"] + df["ask"]) / 2.0
    else:
        mid = df.get("last")
        if mid is None:
            mid = df.get("price")
        if mid is None:
            mid = df.get("bid", df.get("ask", None))
        if mid is None:
            return False, "cannot infer mid price from MT5 ticks", None

    out_dir = (DATA / "ticks")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{symbol}_MT5_{to_utc.strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame({"time_utc": ts, "mid": mid}).to_csv(fname, index=False)
    return True, None, str(fname)

def _mt5_fetch_via_subprocess(cfg: dict):
    """
    Fetch by calling an external collector script with a different Python
    (e.g., .venv-mt5). Requires MT5_PYTHON and MT5_SCRIPT in .env.
    The collector should write CSV(s) under data/ticks/ and print a JSON
    with 'file' or 'files'.
    """
    if not MT5_PYTHON or not MT5_SCRIPT:
        return False, "MT5_PYTHON/MT5_SCRIPT not configured in .env", None

    payload = {
        "login": cfg.get("login"),
        "password": cfg.get("password"),
        "server": cfg.get("server"),
        "path": cfg.get("path"),
        "symbol": cfg.get("symbol", "EURUSD"),
        "days": int(cfg.get("days", 1)),
        "out_dir": str(DATA / "ticks"),
    }

    proc = subprocess.run(
        [MT5_PYTHON, "-u", MT5_SCRIPT],
        input=json.dumps(payload),
        text=True,
        cwd=str(ROOT),
        capture_output=True,
    )
    if proc.returncode != 0:
        return False, f"collector failed: {proc.stderr.strip()}", None

    out = (proc.stdout or "").strip()
    # try to parse a JSON line from stdout
    file_path = None
    try:
        for obj in _extract_json_blocks(out):
            if isinstance(obj, dict):
                file_path = obj.get("file") or (obj.get("files") or [None])[0]
                if file_path:
                    break
    except Exception:
        pass

    if not file_path:
        # fallback: just say it ran; the collector should save into data/ticks/
        return True, None, None

    return True, None, file_path

@app.post("/api/mt5/fetch_ticks")
def fetch_mt5_ticks():
    """
    Pull recent ticks and save a CSV under data/ticks/.
    Uses MetaTrader5 package if available (and MT5_USE_SUBPROCESS is false),
    otherwise falls back to calling an external collector specified in .env.
    """
    req = request.get_json(silent=True) or {}
    cfg = _read_mt5_cfg_and_merge(req)

    # persist merged cfg (nice for the UI's "Set MetaTrader5 Info" flow)
    try:
        MT5_CFG.parent.mkdir(parents=True, exist_ok=True)
        MT5_CFG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass

    if (mt5 is not None) and (not MT5_USE_SUBPROCESS):
        ok, err, fname = _mt5_fetch_via_package(cfg)
    else:
        ok, err, fname = _mt5_fetch_via_subprocess(cfg)

    if not ok:
        return jsonify({"ok": False, "error": err}), 500

    return jsonify({"ok": True, "file": fname})


# --- main -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
