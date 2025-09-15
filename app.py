# -*- coding: utf-8 -*-

import os, sys, json, subprocess, math, random
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta, timezone, date
import time
import requests
import pandas as pd
from flask import Response, stream_with_context
import subprocess, json, time


# -----------------------------------------------------------------------------
# Env / paths
# -----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

ROOT   = Path(__file__).parent.resolve()
PY     = sys.executable
PIPE   = str(ROOT / "fx_pipeline.py")
DATA   = ROOT / "data"
MODELS = ROOT / "models"
MT5_CFG = DATA / "mt5_config.json"

DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SERVE_UI = os.getenv("SERVE_UI", "1") == "1"
if SERVE_UI:
    app = Flask(__name__, static_folder="ui/dist", static_url_path="/")
else:
    app = Flask(__name__)
CORS(app)

if SERVE_UI:
    @app.get("/")
    def _index():
        return app.send_static_file("index.html")

# after load_dotenv()
POLYGON_API_KEY = (os.getenv("POLYGON_API_KEY")
                   or os.getenv("POLY_API_KEY")
                   or os.getenv("POLYGON_KEY")
                   or "").strip()
print("POLYGON_API_KEY loaded:", bool(POLYGON_API_KEY))


# Optional MetaTrader 5 (Windows)
try:
    import MetaTrader5 as mt5  # pip install MetaTrader5
except Exception:
    mt5 = None

MT5_PYTHON = os.getenv("MT5_PYTHON")  # alt Python for MT5
MT5_SCRIPT = os.getenv("MT5_SCRIPT")  # path to external collector
MT5_USE_SUBPROCESS = os.getenv("MT5_USE_SUBPROCESS", "").lower() in {"1", "true", "yes"}

# -----------------------------------------------------------------------------
# Universes
# -----------------------------------------------------------------------------
MAJORS = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD"
]
MINORS = [
    "EURGBP","EURJPY","EURCHF","EURAUD","EURNZD","EURCAD",
    "GBPJPY","GBPCHF","GBPAUD","GBPCAD","GBPNZD",
    "AUDJPY","AUDNZD","AUDCAD","AUDCHF",
    "NZDJPY","NZDCAD","NZDCHF",
    "CADJPY","CADCHF",
    "CHFJPY",
]
EXOTICS = [
    "USDTRY","USDMXN","USDZAR","USDSEK","USDNOK","USDPLN","USDHUF",
    "USDHKD","USDSGD","USDTHB","USDINR","USDILS","USDKRW","USDIDR",
]
METALS = ["XAUUSD","XAGUSD"]

UNIVERSES = {
    "majors": MAJORS,
    "minors": MINORS,
    "exotics": EXOTICS,
    "metals": METALS,
}
UNIVERSE_SYNONYMS = {
    "poly_majors": "majors",
    "poly_minors": "minors",
    "poly_exotics": "exotics",
    "poly_metals": "metals",
    "poly_all": "all",
    "allfx": "all",
}

# -----------------------------------------------------------------------------
# Helpers: run fx_pipeline.py and JSON parsing from stdout
# -----------------------------------------------------------------------------
def run_pipe(*args):
    """Run fx_pipeline.py with args; return (ok, stdout, stderr)."""
    cmd = [PY, "-u", PIPE, *args]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return p.returncode == 0, (p.stdout or "").strip(), (p.stderr or "").strip()

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

def _extract_json_blocks(text: str):
    """Collect JSON objects printed in stdout by scanning for blocks."""
    objs, buf, inside = [], [], False
    for line in (text or "").splitlines():
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

def _parse_last_json_blob(s: str):
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{"); end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            pass
    lines = s.splitlines()
    buf = []
    for i in range(len(lines) - 1, -1, -1):
        buf.insert(0, lines[i])
        try:
            return json.loads("\n".join(buf))
        except Exception:
            continue
    return None

def _sse_subprocess(cmd_list):
    """
    Run a subprocess and stream its stdout lines as SSE 'message' events.
    When it finishes, emit an 'done' event with return code and any JSON found.
    """
    # Flush immediately line-by-line
    proc = subprocess.Popen(
        cmd_list,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Optional: accumulate to try to parse JSON at the end
    collected = []

    try:
        # Tell client to retry in case of disconnects
        yield "retry: 2000\n\n"

        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            collected.append(line)
            payload = {"line": line.rstrip()}
            yield f"data: {json.dumps(payload)}\n\n"

        ret = proc.wait()
        text = "".join(collected)
        metrics = _parse_last_json_line(text) or {}
        done = {"returncode": ret, "metrics": metrics}
        yield f"event: done\ndata: {json.dumps(done)}\n\n"

    finally:
        try:
            proc.kill()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Polygon API key management
# -----------------------------------------------------------------------------
POLY_KEY_PATH = DATA / "polygon_key.json"

def get_polygon_api_key() -> str:
    """Prefer env; else saved key in data/polygon_key.json."""
    k = (os.getenv("POLYGON_API_KEY") or "").strip()
    if k:
        return k
    if POLY_KEY_PATH.exists():
        try:
            obj = json.loads(POLY_KEY_PATH.read_text(encoding="utf-8"))
            k2 = (obj.get("api_key") or "").strip()
            if k2:
                return k2
        except Exception:
            pass
    return ""

def set_polygon_api_key(k: str) -> None:
    POLY_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    POLY_KEY_PATH.write_text(json.dumps({"api_key": k.strip()}, indent=2), encoding="utf-8")

@app.post("/api/polygon/credentials")
def api_polygon_credentials():
    key = (request.json or {}).get("api_key", "").strip()
    if not key:
        return jsonify({"ok": False, "error": "Provide api_key"}), 400
    set_polygon_api_key(key)
    os.environ["POLYGON_API_KEY"] = key  # process-local convenience
    return jsonify({"ok": True})

print("POLYGON_API_KEY loaded:", bool(get_polygon_api_key()))

# -----------------------------------------------------------------------------
# Polygon helpers & robust downloader (range, pagination, de-dupe)
# -----------------------------------------------------------------------------
def _polygon_fx_ticker(sym: str) -> str:
    s = (sym or "EURUSD").upper().replace("/", "")
    return f"C:{s}"

def download_polygon_history(
    symbol: str,
    out_dir: Path,
    years: int = 2,
    timespan: str = "minute",
    multiplier: int = 1,
    api_key: str = None,
    rpm: int = 5,
    max_sleep: int = 90,
    start_date: date = None,   # inclusive
    end_date: date = None,     # exclusive (defaults to today)
    canonical_name: str = None
):
    """
    Download Polygon.io aggregated FX bars to CSV.
    - Sends API key via header and query param.
    - Handles pagination + 429 backoff.
    - Supports [start_date, end_date) or 'years' window.
    - Writes into a canonical file and de-dupes on 'ts_utc'.

    Returns: [<csv_path>] or [] if nothing written.
    """
    import csv
    from urllib.parse import urlparse, parse_qs

    key = (api_key
           or os.getenv("POLYGON_API_KEY")
           or os.getenv("POLYGON_KEY")
           or os.getenv("POLY_API_KEY")
           or "").strip()
    if not key:
        raise RuntimeError("Polygon API key not found in body/override/.env")

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sym = str(symbol)
    if not sym.upper().startswith("C:"):
        sym = f"C:{sym.upper().replace('/', '')}"

    today = datetime.now(timezone.utc).date()
    if end_date is None:
        end_date = today
    if start_date is None:
        start_date = end_date - timedelta(days=365 * int(years))
    if start_date >= end_date:
        return []

    fname = canonical_name or f"{sym.replace(':','_')}_{timespan}_{int(multiplier)}.csv"
    outfile = out_dir / fname

    rpm        = max(int(rpm or 5), 1)
    base_sleep = max(int(60 / rpm), 12)
    backoff    = base_sleep

    url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/{int(multiplier)}/{timespan}/{start_date:%Y-%m-%d}/{end_date:%Y-%m-%d}"
    params  = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key}
    headers = {"X-Polygon-API-Key": key, "Authorization": f"Bearer {key}"}

    tmp_rows = []
    with requests.Session() as s:
        while True:
            r = s.get(url, params=params, headers=headers, timeout=30)

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait_s = int(ra) if (ra and str(ra).isdigit()) else backoff
                wait_s = min(max(wait_s, base_sleep), max_sleep)
                time.sleep(wait_s)
                backoff = min(int(backoff * 1.5) + 1, max_sleep)
                continue

            if r.status_code >= 300:
                raise RuntimeError(f"Polygon {r.status_code}: {r.text[:200]}")

            backoff = base_sleep
            data = r.json() or {}

            for row in (data.get("results") or []):
                t_ms = int(row.get("t"))
                ts = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).isoformat()
                tmp_rows.append([
                    ts,
                    row.get("o"), row.get("h"), row.get("l"), row.get("c"),
                    row.get("v"), row.get("vw"), row.get("n"),
                ])

            next_url = data.get("next_url")
            if next_url:
                from urllib.parse import urlparse, parse_qs
                q = parse_qs(urlparse(next_url).query)
                has_key = any(k.lower() == "apikey" for k in q.keys())
                url = next_url
                params = {} if has_key else {"apiKey": key}
                time.sleep(base_sleep)
                continue
            break

    # If nothing fetched, ensure file exists with header
    import csv as _csv
    if not tmp_rows:
        if not outfile.exists():
            with outfile.open("w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["ts_utc","open","high","low","close","volume","vwap","trades"])
        return []

    cols = ["ts_utc","open","high","low","close","volume","vwap","trades"]
    new_df = pd.DataFrame(tmp_rows, columns=cols)
    if outfile.exists():
        old = pd.read_csv(outfile)
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df

    merged = merged.drop_duplicates(subset=["ts_utc"], keep="last").sort_values("ts_utc")
    merged.to_csv(outfile, index=False)
    return [str(outfile)]

# -----------------------------------------------------------------------------
# API: universes for dropdowns
# -----------------------------------------------------------------------------
@app.get("/api/universes")
def api_universes():
    return jsonify({
        "ok": True,
        "universes": {
            "majors": MAJORS,
            "minors": MINORS,
            "exotics": EXOTICS,
            "metals": METALS,
            "all": sorted(set(MAJORS + MINORS + EXOTICS + METALS)),
        }
    })

# -----------------------------------------------------------------------------
# API: Polygon backfill (symbol/symbols/universe/all) with incremental de-dupe
# -----------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Polygon backfill (drop-in)
# ---------------------------------------------------------------------------

# Optional runtime override: you can set this at runtime from another route
# (e.g., /api/polygon/key) if you like. Default to env var if present.
POLYGON_KEY_OVERRIDE = os.getenv("POLYGON_KEY_OVERRIDE") or None

def _resolve_polygon_api_key(body: dict) -> tuple[str | None, str]:
    """
    Priority: body.api_key > POLYGON_KEY_OVERRIDE > env (POLYGON_API_KEY | POLYGON_KEY | POLY_API_KEY).
    Returns (api_key_or_None, source_string).
    """
    body_key = (body.get("api_key") or "").strip()
    if body_key:
        return body_key, "body"
    if POLYGON_KEY_OVERRIDE and POLYGON_KEY_OVERRIDE.strip():
        return POLYGON_KEY_OVERRIDE.strip(), "override"
    env_key = (
        os.getenv("POLYGON_API_KEY")
        or os.getenv("POLYGON_KEY")
        or os.getenv("POLY_API_KEY")
        or ""
    ).strip()
    if env_key:
        return env_key, "env"
    return None, "none"

def _build_wanted_list(body: dict) -> list[str]:
    """
    Produce the list of requested symbols from body:
      - symbol
      - symbols[]
      - all: true
      - universe in UNIVERSES (with UNIVERSE_SYNONYMS support if present)
    Symbols are normalized like 'EUR/USD' -> 'EURUSD'.
    """
    symbol   = body.get("symbol")
    symbols  = body.get("symbols") or []
    universe = (body.get("universe") or "").lower()
    all_     = bool(body.get("all"))

    # normalize universe synonyms if mapping provided
    try:
        if universe in UNIVERSE_SYNONYMS:
            mapped = UNIVERSE_SYNONYMS[universe]
            if mapped == "all":
                all_ = True
                universe = ""
            else:
                universe = mapped
    except NameError:
        # synonyms map not defined; ignore
        pass

    if symbol:
        return [str(symbol).upper().replace("/", "")]
    if symbols:
        return [str(s).upper().replace("/", "") for s in symbols]
    if all_:
        return list({s for u in UNIVERSES.values() for s in u})
    if universe in UNIVERSES:
        return list(UNIVERSES[universe])

    # nothing matched
    return []

def _last_ts_in_ticks(out_dir: Path, sym: str) -> pd.Timestamp | None:
    """
    Scan existing tick CSVs for a symbol and return the max UTC timestamp found.
    Supports both 'ts_utc' and 'time_utc' columns.
    """
    out_dir = Path(out_dir)
    last = None
    # common filename patterns
    patterns = [
        f"*{sym}*.csv",
        f"*C_{sym}*.csv",
        f"*{sym.upper()}*.csv",
    ]
    for pat in patterns:
        for p in out_dir.glob(pat):
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            tcol = None
            if "ts_utc" in df.columns:
                tcol = "ts_utc"
            elif "time_utc" in df.columns:
                tcol = "time_utc"
            if not tcol:
                continue
            ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
            if ts.notna().any():
                m = ts.max()
                if (last is None) or (m > last):
                    last = m
    return last

def _poly_code(sym: str) -> str:
    """Normalize 'EURUSD' → 'C:EURUSD'."""
    s = (sym or "EURUSD").upper().replace("/", "")
    return f"C:{s}"

def _download_flex(
    *,
    poly_symbol: str,
    out_dir: Path,
    years: int,
    timespan: str,
    multiplier: int,
    api_key: str | None,
    rpm: int,
    start_date: date | None = None,
    end_date: date | None = None,
    canonical_name: str | None = None,
):
    """
    Compatibility shim: try calling download_polygon_history with the richest
    signature first, then progressively simpler ones so older functions still work.
    Returns whatever the function returns (usually list[str] of files).
    """
    kwargs_rich = dict(
        years=years,
        timespan=timespan,
        multiplier=multiplier,
        api_key=api_key,
        rpm=rpm,
        start_date=start_date,
        end_date=end_date,
        canonical_name=canonical_name,
    )
    # Attempt richest → simpler (strip unknown kwargs)
    attempts = [
        kwargs_rich,
        {k: v for k, v in kwargs_rich.items() if k not in ("canonical_name",)},
        {k: v for k, v in kwargs_rich.items() if k not in ("canonical_name", "start_date", "end_date")},
        {k: v for k, v in kwargs_rich.items() if k in ("years", "timespan", "multiplier", "api_key", "rpm")},
        {k: v for k, v in kwargs_rich.items() if k in ("years", "timespan", "multiplier")},
        {"years": years},
        {},
    ]

    last_exc = None
    for kw in attempts:
        try:
            return download_polygon_history(poly_symbol, out_dir, **kw)
        except TypeError as e:
            # signature mismatch, try the next shape
            last_exc = e
            continue
    # If we got here, raise the last signature error
    raise last_exc or RuntimeError("download_polygon_history invocation failed")

@app.post("/api/polygon/backfill")
def polygon_backfill():
    body        = request.get_json(silent=True) or {}
    years       = int(body.get("years", 2))
    timespan    = str(body.get("timespan", "minute"))
    multiplier  = int(body.get("multiplier", 1))
    rpm         = max(int(body.get("rpm", 5)), 1)
    incremental = bool(body.get("incremental", True))

    wanted = _build_wanted_list(body)
    if not wanted:
        return jsonify({
            "ok": False,
            "error": "Provide 'symbol', 'symbols', set 'all': true, or 'universe' in majors|minors|exotics|metals"
        }), 400

    api_key, key_source = _resolve_polygon_api_key(body)
    if not api_key:
        # Don’t hard-fail; the downloader will raise per-symbol if truly required
        print("[WARN] Polygon API key not found (body/override/env). Requests may 401.")

    out_dir = (DATA / "ticks")
    out_dir.mkdir(parents=True, exist_ok=True)

    # inter-symbol throttle derived from rpm
    sleep_s = max(int(60 / rpm), 12)

    downloaded: list[dict] = []
    errors: dict[str, str] = {}

    now_utc = datetime.now(timezone.utc)
    for i, sym in enumerate(wanted, start=1):
        poly = _poly_code(sym)

        # incremental start: resume from the last timestamp seen
        start_date = None
        end_date   = None
        canon_name = f"{poly.replace(':', '_')}_{timespan}_{multiplier}.csv"

        if incremental:
            last_ts = _last_ts_in_ticks(out_dir, sym)
            if last_ts is not None:
                # back 5 minutes to overlap; downloader should de-dup on write
                start_date = (last_ts - pd.Timedelta(minutes=5)).date()
                # safety: never set start_date after "today"
                if start_date > now_utc.date():
                    start_date = now_utc.date()

        try:
            files = _download_flex(
                poly_symbol=poly,
                out_dir=out_dir,
                years=years,
                timespan=timespan,
                multiplier=multiplier,
                api_key=api_key,
                rpm=rpm,
                start_date=start_date,
                end_date=end_date,
                canonical_name=canon_name,
            )
            downloaded.append({"symbol": sym, "files": files or []})
        except Exception as e:
            errors[sym] = str(e)

        # throttle between symbols to respect rpm target
        if i < len(wanted):
            time.sleep(sleep_s)

    return jsonify({
        "ok": True,
        "key_source": key_source,   # body | override | env | none (for debugging)
        "years": years,
        "timespan": timespan,
        "multiplier": multiplier,
        "rpm": rpm,
        "sleep_s": sleep_s,
        "incremental": incremental,
        "count": len(wanted),
        "downloaded": downloaded,
        "errors": errors,
    })


# -----------------------------------------------------------------------------
# Ticks feed for 48h chart
# -----------------------------------------------------------------------------

@app.post("/api/polygon/key")
def api_polygon_key():
    """
    Optionally set a runtime Polygon key without editing .env.
    (Not required if your .env is already set.)
    """
    global POLYGON_KEY_OVERRIDE
    body = request.get_json(silent=True) or {}
    key = (body.get("api_key") or "").strip()
    if not key:
        return jsonify({"ok": False, "error": "No api_key provided"}), 400
    POLYGON_KEY_OVERRIDE = key
    return jsonify({"ok": True})

@app.get("/api/ticks")
def api_ticks_48h():
    symbol = (request.args.get("symbol") or "EURUSD").upper().replace("/", "")
    hours  = int(request.args.get("hours", 48))

    root = DATA / "ticks"
    if not root.exists():
        return jsonify({"ok": False, "error": "ticks folder missing"}), 404

    files = sorted(list(root.glob(f"*{symbol}*.csv")))
    if not files:
        return jsonify({"ok": True, "symbol": symbol, "points": []})

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    frames = []

    for p in files[-6:]:  # last few files
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        tcol = "ts_utc" if "ts_utc" in df.columns else ("time_utc" if "time_utc" in df.columns else None)
        vcol = "close"  if "close"  in df.columns else ("mid"      if "mid"      in df.columns else None)
        if not tcol or not vcol:
            continue
        df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        df = df.dropna(subset=[tcol])
        df = df[df[tcol] >= cutoff][[tcol, vcol]]
        if not df.empty:
            frames.append(df.rename(columns={tcol: "ts_utc", vcol: "price"}))

    if not frames:
        return jsonify({"ok": True, "symbol": symbol, "points": []})

    out = pd.concat(frames).sort_values("ts_utc").drop_duplicates("ts_utc")
    out = out.tail(15000)  # cap payload
    points = [{"t": ts.isoformat(), "v": float(v)} for ts, v in zip(out["ts_utc"], out["price"])]
    return jsonify({"ok": True, "symbol": symbol, "points": points})

def _read_ticks_any(symbol: str) -> pd.DataFrame:
    sym = symbol.upper()
    tick_root = DATA / "ticks"
    patterns = [f"*{sym}*.csv", f"*C_{sym}*.csv", f"{sym}*.csv"]  # broaden like /api/chart
    files = []
    for pat in patterns:
        files.extend(tick_root.glob(pat))
    if not files:
        return pd.DataFrame(columns=["mid"])

    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            if {"ts_utc","close"}.issubset(df.columns):
                ts = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
                mid = pd.to_numeric(df.get("close", df.get("c")), errors="coerce")
            elif {"time_utc","mid"}.issubset(df.columns):
                ts  = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
                mid = pd.to_numeric(df["mid"], errors="coerce")
            else:
                continue
            f = pd.DataFrame({"mid": mid.to_numpy()}, index=ts).dropna().sort_index()
            frames.append(f[~f.index.duplicated(keep="last")])
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["mid"])
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


@app.get("/api/ticks/recent")
def api_ticks_recent():
    """
    GET /api/ticks/recent?symbol=EURUSD&hours=48
    Downsamples to 1-minute points for charting.
    """
    symbol = (request.args.get("symbol") or request.args.get("ticker") or "EURUSD").upper().replace("/", "")
    hours  = int(request.args.get("hours", 48))

    df = _read_ticks_any(symbol)
    if df.empty:
        return jsonify({"ok": True, "symbol": symbol, "points": [], "from": None, "to": None})

    now_utc = pd.Timestamp.now(tz="UTC")
    cutoff  = now_utc - pd.Timedelta(hours=hours)

    # Keep only recent window, 1-min last price
    df = df[df.index >= cutoff]
    m1 = df["mid"].resample("min").last().dropna()

    if m1.empty:
        return jsonify({"ok": True, "symbol": symbol, "points": [], "from": cutoff.isoformat(), "to": now_utc.isoformat()})

    pts = [{"t": ts.isoformat(), "mid": float(v)} for ts, v in m1.items()]
    return jsonify({
        "ok": True,
        "symbol": symbol,
        "points": pts,
        "from": m1.index.min().isoformat(),
        "to":   m1.index.max().isoformat(),
        "count": int(len(pts)),
    })


@app.get("/api/predict")
def api_predict_get():
    objective = request.args.get("objective", "acc")
    return jsonify(_do_predict(objective))

@app.get("/api/chart")
def api_chart():
    """Return last N hours of minute bars for a ticker for charting."""
    from pandas import to_datetime, Timestamp, Timedelta

    ticker = (request.args.get("ticker") or request.args.get("symbol") or "EURUSD").upper().replace("/", "")
    hours  = int(request.args.get("hours", 48))

    tick_dir = DATA / "ticks"
    if not tick_dir.exists():
        return jsonify({"ok": True, "ticker": ticker, "points": []})

    # Search broadly so we match Polygon (C_EURUSD...) and MT5 (EURUSD_...)
    files = []
    for pat in (f"*{ticker}*.csv", f"*C_{ticker}*.csv", f"{ticker}*.csv"):
        files.extend(tick_dir.glob(pat))
    files = sorted(set(files))
    if not files:
        # return ok with empty points instead of 404 — UI can show "no data"
        return jsonify({"ok": True, "ticker": ticker, "points": []})

    parts = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}

        # Normalize to index=UTC and one 'mid' column
        if "time_utc" in cols:  # MT5/simple
            ts = to_datetime(df[cols["time_utc"]], utc=True, errors="coerce")
            price = pd.to_numeric(df.get("mid", df.get("close", df.get("c"))), errors="coerce")
        elif "ts_utc" in cols:  # Polygon aggs
            ts = to_datetime(df[cols["ts_utc"]], utc=True, errors="coerce")
            price = pd.to_numeric(df.get("close", df.get("c")), errors="coerce")
        else:
            continue

        sub = pd.DataFrame({"mid": price}, index=ts).dropna().sort_index()
        sub = sub[~sub.index.duplicated(keep="last")]
        if not sub.empty:
            parts.append(sub)

    if not parts:
        return jsonify({"ok": True, "ticker": ticker, "points": []})

    ticks = pd.concat(parts).sort_index()
    ticks = ticks[~ticks.index.duplicated(keep="last")]

    now = Timestamp.now(tz="UTC")
    cutoff = now - Timedelta(hours=hours)
    ticks = ticks.loc[ticks.index >= cutoff]

    # 1-minute resample (FutureWarning-safe: use 'min' not 'T')
    series = ticks["mid"].resample("min").last().dropna()

    points = [{"t": ts.isoformat(), "mid": float(val)} for ts, val in series.items()]
    return jsonify({"ok": True, "ticker": ticker, "from": cutoff.isoformat(), "to": now.isoformat(), "points": points})


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return jsonify({"ok": True})

# -----------------------------------------------------------------------------
# Build / Train / Predict / Holdout
# -----------------------------------------------------------------------------
@app.post("/api/build")
def api_build():
    body = request.get_json(silent=True) or {}
    ticks_glob = body.get("ticks_glob") or body.get("glob") or "data/ticks/*.csv"
    sym = (body.get("symbol") or "").strip().upper().replace("/", "")
    if sym:
        canonical = f"data/ticks/C_{sym}_*.csv"
        alt = f"data/ticks/*{sym}*.csv"
        if any(Path().glob(canonical)):
            ticks_glob = canonical
        else:
            ticks_glob = alt
    ok, out, err = run_pipe("build", "--ticks-glob", ticks_glob)
    return jsonify({"ok": ok, "stdout": out, "stderr": err, "glob": ticks_glob})

@app.post("/api/train")
def api_train():
    body = request.get_json(silent=True) or {}
    symbol = (body.get("symbol") or request.args.get("symbol") or "").upper().replace("/", "")
    args = ["train"]
    if symbol:
        args += ["--symbol", symbol]
    ok, out, err = run_pipe(*args)
    metrics = _parse_last_json_line(out)
    return jsonify({"ok": ok, "metrics": metrics, "stdout": out, "stderr": err})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(silent=True) or {}
    symbol = (body.get("symbol") or request.args.get("symbol") or request.args.get("ticker") or "").upper().replace("/", "")
    objective = request.args.get("objective", "acc")

    args = ["predict", "--objective", objective]
    if symbol:
        args += ["--symbol", symbol]  # will be ignored if fx_pipeline doesn't support it

    ok, out, err = run_pipe(*args)
    preds = _parse_last_json_blob(out)
    return jsonify({
        "ok": ok,
        "symbol": symbol or None,
        "objective": objective,
        "predictions": preds,
        "stdout": out,
        "stderr": err
    })


@app.post("/api/holdout")
def api_holdout():
    body = request.get_json(silent=True) or {}
    cutoff = body.get("cutoff")
    last_days = body.get("last_days")
    args = ["holdout"]
    if cutoff:
        args += ["--cutoff", str(cutoff)]
    elif last_days:
        args += ["--last-days", str(int(last_days))]
    else:
        return jsonify({"ok": False, "error": "Provide cutoff or last_days"}), 400

    ok, out, err = run_pipe(*args)
    res = _parse_last_json_line(out)
    return jsonify({"ok": ok, "result": res, "stdout": out, "stderr": err})



def _sse_pack(data: dict, event: str | None = None) -> str:
    """
    Format a dict as a Server-Sent Event (SSE) frame.
    """
    out = []
    if event:
        out.append(f"event: {event}")
    out.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    out.append("")  # blank line ends the event
    return "\n".join(out)

def _stream_process(cmd: list[str], parse_last_json: bool = False, done_event: str = "done"):
    """
    Run a subprocess and stream its stdout lines as SSE.
    Optionally parse the last JSON-ish object from the full buffer.
    """
    buf = []
    p = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in iter(p.stdout.readline, ""):
            line = line.rstrip("\n")
            buf.append(line)
            yield _sse_pack({"line": line})
        p.wait()
        payload = {"returncode": p.returncode}
        if parse_last_json:
            try:
                payload["metrics"] = _parse_last_json_line("\n".join(buf))
            except Exception:
                pass
        yield _sse_pack(payload, event=done_event)
    finally:
        try:
            p.stdout.close()
        except Exception:
            pass

@app.get("/api/stream/build")
def stream_build():
    """
    SSE stream for 'build' (dataset creation from ticks).
    Accepts ?ticks_glob=... (default data/ticks/*.csv).
    """
    ticks_glob = request.args.get("ticks_glob", "data/ticks/*.csv")
    cmd = [PY, "-u", PIPE, "build", "--ticks-glob", ticks_glob]
    return Response(
        stream_with_context(_stream_process(cmd, parse_last_json=False, done_event="done")),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/api/stream/train")
def stream_train():
    """
    SSE stream for 'train'.
    Optionally accepts ?symbol=EURUSD (if your fx_pipeline supports it; else ignored).
    """
    symbol = request.args.get("symbol")
    cmd = [PY, "-u", PIPE, "train"]
    if symbol:
        cmd += ["--symbol", symbol]
    return Response(
        stream_with_context(_stream_process(cmd, parse_last_json=True, done_event="done")),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# --- simple prediction series over the last N hours (flat overlay) -----------
@app.get("/api/predict/series")
def api_predict_series():
    """
    GET /api/predict/series?symbol=EURUSD&hours=24&objective=acc[&align_to_ticks=1]
    Returns per-window probability repeated across each minute in the chosen window.
    If align_to_ticks=1, the window ends at the last tick time for that symbol.
    """
    symbol    = (request.args.get("symbol") or "EURUSD").upper()
    hours     = int(request.args.get("hours", 24))
    objective = request.args.get("objective", "acc")
    align     = request.args.get("align_to_ticks", "0") in {"1", "true", "yes"}

    # run pipeline (per-symbol)
    args = ["predict", "--objective", objective]
    if symbol:
        args += ["--symbol", symbol]
    ok, out, err = run_pipe(*args)
    preds = _parse_last_json_blob(out) or {}

    # choose window end
    end_utc = pd.Timestamp.now(tz="UTC")
    if align:
        ticks = _read_ticks_any(symbol)
        if not ticks.empty:
            last_tick = ticks.index.max()
            # guard: if last tick is “recent enough” (<= 7d old), snap end to it
            if pd.notna(last_tick) and (end_utc - last_tick) <= pd.Timedelta(days=7):
                end_utc = last_tick.tz_convert("UTC") if last_tick.tz is not None else last_tick.tz_localize("UTC")

    start_utc = end_utc - pd.Timedelta(hours=hours)
    idx = pd.date_range(start=start_utc, end=end_utc, freq="min", tz="UTC")

    series = {}
    for key in ("daily", "2_10", "w2_10", "w3_12"):
        p = preds.get(key)
        if not p:
            continue
        prob = p.get("proba_eff", p.get("proba_raw"))
        if prob is None:
            continue
        out_key = "2_10" if key == "w2_10" else key
        series[out_key] = [{"t": ts.isoformat(), "proba": float(prob)} for ts in idx]

    return jsonify({
        "ok": True,
        "symbol": symbol,
        "objective": objective,
        "hours": hours,
        "aligned_end": align,
        "from": start_utc.isoformat(),
        "to": end_utc.isoformat(),
        "series": series,
        "snapshot": preds,
        "stderr": err,
    })





# -----------------------------------------------------------------------------
# MetaTrader 5: config + fetch
# -----------------------------------------------------------------------------
@app.post("/api/mt5/config")
def save_mt5_config():
    MT5_CFG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "login":    (request.json or {}).get("login", ""),
        "password": (request.json or {}).get("password", ""),
        "server":   (request.json or {}).get("server", ""),
        "path":     (request.json or {}).get("path", ""),
        "symbol":   (request.json or {}).get("symbol", "EURUSD"),
        "days":     int((request.json or {}).get("days", 1)),
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
    for k in ["login", "password", "server", "path", "symbol", "days"]:
        v = (req_json or {}).get(k)
        if v is not None:
            cfg[k] = v
    cfg.setdefault("symbol", "EURUSD")
    cfg["days"] = int(cfg.get("days") or 1)
    if cfg.get("login"):
        try:
            cfg["login"] = int(cfg["login"])
        except Exception:
            pass
    return cfg

def _mt5_fetch_via_package(cfg: dict):
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

    to_utc = datetime.now(timezone.utc)
    from_utc = to_utc - timedelta(days=days)
    ticks = mt5.copy_ticks_range(symbol, from_utc, to_utc, mt5.COPY_TICKS_ALL)
    mt5.shutdown()

    if ticks is None or len(ticks) == 0:
        return False, "no ticks returned", None

    df = pd.DataFrame(ticks)
    if "time_msc" in df.columns and df["time_msc"].notna().any():
        ts = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    else:
        ts = pd.to_datetime(df["time"], unit="s", utc=True)

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
        return True, None, None
    return True, None, file_path

@app.post("/api/mt5/fetch_ticks")
def fetch_mt5_ticks():
    """Pull recent ticks and save a CSV under data/ticks/."""
    req = request.get_json(silent=True) or {}
    cfg = _read_mt5_cfg_and_merge(req)
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

# --- Policy management (drop in) -------------------------------------------
def _policy_path(wkey: str) -> Path:
    return MODELS / f"policy_{wkey}.json"

def _default_policy():
    return {
        "flip": False,
        "active": True,
        "acc": {"threshold": 0.5},
        "mcc": {"threshold": 0.5},
        "pnl": {"threshold": 0.5},
    }

def _read_policy(wkey: str) -> dict:
    p = _policy_path(wkey)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return _default_policy()

@app.get("/api/policies")
def get_policies():
    out = {k: _read_policy(k) for k in ["daily", "w2_10", "w3_12"]}
    return jsonify({"ok": True, "policies": out})

@app.post("/api/policies")
def set_policies():
    body = request.get_json(silent=True) or {}
    updated = {}
    for k in ["daily", "w2_10", "w3_12"]:
        if k not in body:
            continue
        cur = _read_policy(k)
        inc = body[k] or {}

        # booleans
        for key in ("flip", "active"):
            if key in inc:
                cur[key] = bool(inc[key])

        # thresholds
        for key in ("acc", "mcc", "pnl"):
            if key in inc and isinstance(inc[key], dict):
                thr = inc[key].get("threshold")
                if thr is not None:
                    cur.setdefault(key, {})["threshold"] = float(thr)

        _policy_path(k).write_text(json.dumps(cur, indent=2), encoding="utf-8")
        updated[k] = cur

    return jsonify({"ok": True, "policies": updated})


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
