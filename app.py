# app.py
import os
import atexit
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

import fx_pipeline as fx  # your single-file pipeline

# ---- config from env (so Docker can mount /app/data, /app/models)
def cfg_from_env():
    cfg = fx.DEFAULT_CONFIG.copy()
    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))
    models_dir = Path(os.getenv("MODELS_DIR", "/app/models"))

    cfg["paths"]["ticks_glob"]    = os.getenv("TICKS_GLOB", str(data_dir / "ticks/*.csv"))
    cfg["paths"]["features_out"]  = str(data_dir / "features.parquet")
    cfg["paths"]["feats_only"]    = str(data_dir / "features_only.parquet")
    cfg["paths"]["labs_only"]     = str(data_dir / "labels_only.parquet")
    cfg["paths"]["models_dir"]    = str(models_dir)

    # make sure directories exist
    Path(cfg["paths"]["models_dir"]).mkdir(parents=True, exist_ok=True)
    Path(data_dir, "ticks").mkdir(parents=True, exist_ok=True)
    return cfg

app = Flask(__name__, static_folder="static", static_url_path="/")
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True

# ---------- Static UI ----------
@app.route("/")
def index():
    # Your existing index.html should live at ./static/index.html
    return send_from_directory(app.static_folder, "index.html")

# ---------- Health ----------
@app.get("/api/health")
def health():
    return jsonify(ok=True)

# ---------- Predict ----------
@app.get("/api/predict")
def api_predict():
    objective = request.args.get("objective", "acc")  # acc|mcc|pnl
    cfg = cfg_from_env()
    features_p = Path(cfg["paths"]["features_out"])
    models_dir = Path(cfg["paths"]["models_dir"])

    # ensure we have something to predict on
    if not features_p.exists():
        # try salvage checkpoints, else full build
        ds = fx.rebuild_from_checkpoints(cfg)
        if ds is None:
            _, _, _ = fx.build_all(cfg)
            df2 = pd.read_parquet(features_p)
            fx.train_and_eval_per_window(df2, cfg, models_dir)
            fx.build_and_save_policies(df2, cfg, models_dir)

    df = pd.read_parquet(features_p)
    feats_for_pred = df.drop(columns=[c for c in df.columns if c.endswith("_pips") or c.endswith("_up")])
    preds = fx.predict_next_day(feats_for_pred, cfg, models_dir, objective=objective)
    return jsonify(preds)

# ---------- Manual retrain ----------
@app.post("/api/retrain")
def api_retrain():
    """
    Daily job also calls this logic. You or the client can POST here
    to force a rebuild+train now.
    """
    cfg = cfg_from_env()
    models_dir = Path(cfg["paths"]["models_dir"])

    # 1) (Optional) fetch new ticks from your data provider API and write CSV into data/ticks/
    #    Implement this if you have a provider; otherwise just drop new CSVs into /data/ticks/
    # fetch_new_ticks_from_api(cfg)  # <-- implement if you want automated fetching

    # 2) Build dataset (salvage if checkpoints exist)
    ds = fx.rebuild_from_checkpoints(cfg)
    if ds is None:
        _, _, ds = fx.build_all(cfg)

    # 3) Train + save policies
    df = pd.read_parquet(cfg["paths"]["features_out"])
    metrics = fx.train_and_eval_per_window(df, cfg, models_dir)
    policies = fx.build_and_save_policies(df, cfg, models_dir)

    return jsonify(metrics=metrics, policies=policies)

# ---------- Tick upload (optional UI path) ----------
@app.post("/api/upload-ticks")
def upload_ticks():
    """
    Let the client upload a CSV of ticks via the UI.
    The file is just saved to /app/data/ticks and will be included next retrain.
    """
    if "file" not in request.files:
        return jsonify(error="no file"), 400
    f = request.files["file"]
    fname = f.filename or "ticks.csv"
    save_path = Path(os.getenv("DATA_DIR", "/app/data")) / "ticks" / fname
    save_path.parent.mkdir(parents=True, exist_ok=True)
    f.save(save_path)
    return jsonify(saved=str(save_path))

# in app.py, near the top
def fetch_new_ticks_from_api(cfg):
    """
    Placeholder for in-container fetching.
    MetaTrader5 is Windows-only, so we do not use it here.
    Keep this as a NO-OP; the Windows collector writes CSVs to /app/data/ticks.
    """
    return


# ---------- Daily scheduler inside the container ----------
def _daily_retrain_job():
    try:
        cfg = cfg_from_env()
        models_dir = Path(cfg["paths"]["models_dir"])

        # (Optionally) fetch & append new ticks from API here
        # fetch_new_ticks_from_api(cfg)

        ds = fx.rebuild_from_checkpoints(cfg)
        if ds is None:
            _, _, ds = fx.build_all(cfg)

        df = pd.read_parquet(cfg["paths"]["features_out"])
        fx.train_and_eval_per_window(df, cfg, models_dir)
        fx.build_and_save_policies(df, cfg, models_dir)
        print("[scheduler] retrain complete", flush=True)
    except Exception as e:
        print(f"[scheduler] error: {e}", flush=True)

if os.getenv("ENABLE_SCHEDULER", "1") == "1":
    scheduler = BackgroundScheduler(timezone="UTC")
    hour = int(os.getenv("SCHEDULE_UTC_HOUR", "06"))     # default 06:10 UTC daily
    minute = int(os.getenv("SCHEDULE_UTC_MINUTE", "10"))
    scheduler.add_job(_daily_retrain_job, "cron", hour=hour, minute=minute, id="daily_retrain", replace_existing=True)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown(wait=False))

# Gunicorn needs this symbol:
application = app  # alias

