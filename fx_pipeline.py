# -*- coding: utf-8 -*-
"""
fx_pipeline.py
Single-file FX pipeline:
- Robust CSV ingest (→ UTC index + 'mid')
- Per-window bars/features (daily, w2_10, w3_12)
- Train/eval with manifest-locked columns
- Threshold policies (acc/mcc/pnl)
- Predict next session/day
- Polygon.io historical downloader (minute bars)

Python 3.8+
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import numpy as np
import pandas as pd
import requests
from joblib import dump, load
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================== Polygon downloader ===========================

def _poly_fx_ticker(sym: str) -> str:
    # Polygon FX tickers look like C:EURUSD
    return f"C:{(sym or 'EURUSD').upper().replace('/', '')}"

def _append_api_key(url: str, api_key: str) -> str:
    """Ensure ?apiKey=... is present in the url (used for next_url pages)."""
    pu = urlparse(url)
    q = parse_qs(pu.query)
    q["apiKey"] = [api_key]  # force/replace
    # flatten singletons for urlencode
    q_flat = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in q.items()}
    return urlunparse((pu.scheme, pu.netloc, pu.path, pu.params, urlencode(q_flat), pu.fragment))

def download_polygon_history(
    symbol,
    out_dir,
    years: int = 2,
    timespan: str = "minute",
    multiplier: int = 1,
    api_key: Optional[str] = None,
    rpm: int = 5,
):
    """
    Download Polygon.io aggregated bars for an FX symbol into CSV under out_dir.
    Returns a LIST with the written CSV path (or empty if nothing).
    Accepts both "EURUSD" and "C:EURUSD".
    """
    api_key = (api_key or os.getenv("POLYGON_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set (env) and 'api_key' not provided")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize to Polygon FX ticker
    if not str(symbol).upper().startswith("C:"):
        symbol = _poly_fx_ticker(symbol)

    # Date range
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=365 * int(years))

    outfile = out_dir / f"{symbol.replace(':','_')}_{timespan}_{int(multiplier)}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    wrote_any = False

    # Base URL + params (include apiKey)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{int(multiplier)}/{timespan}/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}

    # Send key in headers too (some proxies drop query or headers; we do both)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Polygon-API-Key": api_key,
    }

    # throttle target (free plan ~5 rpm)
    rpm = max(int(rpm or 5), 1)
    sleep_s = max(int(60 / rpm), 12)

    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts_utc", "open", "high", "low", "close", "volume", "vwap", "trades"])

        while True:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(sleep_s)
                continue
            if r.status_code >= 300:
                raise RuntimeError(f"Polygon {r.status_code}: {r.text[:200]}")

            data = r.json() or {}
            for row in data.get("results", []):
                t_ms = int(row.get("t"))
                ts = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).isoformat()
                w.writerow([
                    ts,
                    row.get("o"), row.get("h"), row.get("l"), row.get("c"),
                    row.get("v"), row.get("vw"), row.get("n"),
                ])
                wrote_any = True

            next_url = data.get("next_url")
            if next_url:
                # Be explicit: add apiKey to next_url and keep headers
                url = _append_api_key(next_url, api_key)
                params = {}  # next_url contains query already
                time.sleep(sleep_s)
                continue
            break

    return [str(outfile)] if wrote_any else []


# ============================== Default Config ==============================

DEFAULT_CONFIG = {
    "pip_size": 0.0001,
    "pt_timezone": "America/Los_Angeles",
    "paths": {
        "ticks_glob": "data/ticks/*.csv",
        "econ_calendar": "data/econ_calendar.csv",
        "features_out": "data/features.parquet",
        "models_dir": "models",
        "feats_only": "data/features_only.parquet",   # optional salvage
        "labs_only": "data/labels_only.parquet",      # optional salvage
    },
    "windows": {
        "daily": {"start": "00:00", "end": "23:59"},
        "w2_10": {"start": "02:00", "end": "10:00"},
        "w3_12": {"start": "03:00", "end": "12:00"},
    },
    "round_levels_pips": [10, 25, 50, 100],
    "round_proximity_thresh_pips": 5,
    "breakout": {
        "lookback_sessions": 10,
        "atr_window": 20,
        "breakout_gamma": 0.25,
        "fail_backoff_pips": 2,
    },
    "exhaustion": {"ma_window": 20, "atr_window": 20},
    "tick_volume": {"baseline_days": 20, "hot_threshold": 1.5},
    "pre_news": {
        "pre_window_minutes": 180,
        "post_window_minutes": 60,
        "impacts": ["HIGH"],
        "currencies": ["USD", "EUR"],
    },
    "cv": {"n_splits": 5, "embargo_days": 1, "test_size_days": 60},
}

# ============================== Config Utils ================================

def load_config(path: Optional[str]) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep-ish copy
    if path:
        p = Path(path)
        if p.exists():
            try:
                import yaml  # optional
                with p.open("r", encoding="utf-8") as f:
                    user = yaml.safe_load(f) or {}
                for k, v in user.items():
                    if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                        cfg[k].update(v)
                    else:
                        cfg[k] = v
            except Exception:
                # If PyYAML not installed or parse error, just ignore overrides
                pass
    return cfg

def ensure_dir(d: str | Path) -> Path:
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_manifests_from_dataset(df: pd.DataFrame, models_dir: Path) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for wkey in ["daily", "w2_10", "w3_12"]:
        p = models_dir / f"cols_{wkey}.json"
        if p.exists():
            continue
        if f"is_{wkey}" not in df.columns:
            continue
        feat_cols = (
            [c for c in df.columns if c.startswith(f"{wkey}_") and not (c.endswith("_up") or c.endswith("_pips"))]
            + [c for c in df.columns if c.startswith("dow_")]
            + ["is_eom", "is_eoq", f"is_{wkey}"]
        )
        seen = set()
        feat_cols = [c for c in feat_cols if not (c in seen or seen.add(c))]
        p.write_text(json.dumps(feat_cols), encoding="utf-8")


# ============================== CSV Ingestion ===============================

def _norm_col(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _read_csv_any_sep(path: Path, header="infer") -> pd.DataFrame:
    try:
        return pd.read_csv(path, engine="python", sep=None, header=0 if header == "infer" else None)
    except Exception:
        pass
    for sep in [",", ";", "\t", "|"]:
        try:
            return pd.read_csv(path, sep=sep, header=0 if header == "infer" else None)
        except Exception:
            continue
    return pd.read_csv(path, header=0 if header == "infer" else None)

def _infer_server_offset_from_filename(path: Path) -> tuple[int, bool]:
    name = path.name.upper()
    m = re.search(r"GMT([+-]\d+)", name)
    base = int(m.group(1)) if m else 0
    has_us_dst = "US-DST" in name or "USDST" in name or "US_DST" in name
    return base, has_us_dst

def _us_dst_active(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    yrs = ts.dt.year
    starts, ends = {}, {}
    for y in yrs.dropna().unique():
        d = pd.Timestamp(year=y, month=3, day=1)
        first_sun = d + pd.Timedelta(days=(6 - d.weekday()) % 7)
        start = first_sun + pd.Timedelta(weeks=1, hours=2)
        d2 = pd.Timestamp(year=y, month=11, day=1)
        first_sun_nov = d2 + pd.Timedelta(days=(6 - d2.weekday()) % 7)
        end = first_sun_nov + pd.Timedelta(hours=2)
        starts[y], ends[y] = start, end
    out = pd.Series(False, index=ts.index)
    for y, st in starts.items():
        en = ends[y]
        mask = (yrs == y)
        out.loc[mask] = (ts.loc[mask] >= st) & (ts.loc[mask] < en)
    return out

_date_only_re = re.compile(r"^\d{4}[./-]\d{1,2}[./-]\d{1,2}$")
_time_only_re = re.compile(r"^\d{1,2}:\d{2}:\d{2}(?:\.\d{1,6})?$")

def _series_is_numeric(s: pd.Series, frac: float = 0.8) -> bool:
    return pd.to_numeric(s, errors="coerce").notna().mean() >= frac

def _detect_datetime_column(df: pd.DataFrame) -> Optional[pd.Series]:
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    cols_norm = {_norm_col(c): c for c in df.columns}
    keys = list(cols_norm.keys())

    direct = [
        "timestamp", "datetime", "date_time", "date time", "gmt time", "utc time",
        "time gmt", "time (gmt)", "time (utc)", "local time", "bar time", "bar start time", "time", "date",
    ]
    for k in direct:
        if k in cols_norm:
            col = df[cols_norm[k]].astype(str)
            if k == "time":
                sample = col.head(50)
                if sample.str.contains(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}", regex=True).mean() >= 0.5:
                    return col
            else:
                return col

    date_key = next((k for k in keys if k.startswith("date")), None)
    time_key = next((k for k in keys if k.startswith("time") or " time" in k), None)
    if date_key and time_key:
        return (df[cols_norm[date_key]].astype(str) + " " + df[cols_norm[time_key]].astype(str))

    # epoch seconds/ms
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.8:
            if ((s > 1_000_000_000) & (s < 4_000_000_000)).mean() > 0.8:
                return s.astype("Int64").astype(str)
            if ((s > 1_000_000_000_000) & (s < 4_000_000_000_000)).mean() > 0.8:
                return s.astype("Int64").astype(str)

    for c in df.columns:
        sample = pd.to_datetime(df[c], errors="coerce")
        if sample.notna().mean() >= 0.8:
            return df[c].astype(str)

    return None

def read_tick_csv(path: Path) -> pd.DataFrame:
    """
    Return DataFrame indexed by UTC timestamp with a single 'mid' column.
    Supports headered and headless CSVs.
    """
    # Try headered
    df = _read_csv_any_sep(path, header="infer")
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    ts_text = _detect_datetime_column(df)
    headerless = ts_text is None

    def _finalize(ts_text: pd.Series, mid: pd.Series) -> pd.DataFrame:
        ts_num = pd.to_numeric(ts_text, errors="coerce")
        if ts_num.notna().mean() > 0.8:
            is_ms = (ts_num > 1_000_000_000_000).mean() > 0.5
            ts_naive = pd.to_datetime(ts_num, unit=("ms" if is_ms else "s"), errors="coerce")
        else:
            ts_naive = pd.to_datetime(ts_text, errors="coerce")
        if ts_naive.isna().all():
            raise ValueError(f"{path} has unparseable timestamps")

        base_off, has_us_dst = _infer_server_offset_from_filename(path)
        if base_off != 0 or has_us_dst:
            if has_us_dst:
                dst_active = _us_dst_active(ts_naive)
                total_off = base_off + dst_active.astype(int)
            else:
                total_off = pd.Series(base_off, index=ts_naive.index)
            ts_utc = pd.DatetimeIndex(ts_naive - pd.to_timedelta(total_off, unit="h"), tz="UTC")
        else:
            ts_utc = pd.DatetimeIndex(ts_naive, tz="UTC")

        out = pd.DataFrame({"mid": pd.to_numeric(mid, errors="coerce").to_numpy()}, index=ts_utc)
        out = out[np.isfinite(out["mid"])]
        out = out[~out.index.duplicated(keep="first")].sort_index()
        return out

    if headerless:
        df = _read_csv_any_sep(path, header=None)
        df.columns = [f"col{i}" for i in range(df.shape[1])]
        if df.empty:
            raise ValueError(f"{path} is empty")

        ts0 = str(df["col0"].iloc[0])
        if df.shape[1] >= 2 and _date_only_re.match(ts0) and _time_only_re.match(str(df["col1"].iloc[0])):
            ts_text = (df["col0"].astype(str) + " " + df["col1"].astype(str))
            first_price = 2
        else:
            ts_text = df["col0"].astype(str)
            first_price = 1

        mid = None
        if df.shape[1] >= first_price + 2:
            p1 = pd.to_numeric(df[f"col{first_price}"], errors="coerce")
            p2 = pd.to_numeric(df[f"col{first_price+1}"], errors="coerce")
            if p1.notna().mean() > 0.5 and p2.notna().mean() > 0.5:
                mid = (p1 + p2) / 2.0
        if mid is None and df.shape[1] >= first_price + 1:
            p = pd.to_numeric(df[f"col{first_price}"], errors="coerce")
            if p.notna().mean() > 0.5:
                mid = p
        if mid is None:
            raise ValueError(f"{path} headerless parse failed (no price columns)")

        return _finalize(ts_text, mid)

    # Headered → map to mid
    lower_map = {_norm_col(c): c for c in df.columns}
    mid: Optional[pd.Series] = None
    if "mid" in lower_map:
        mid = pd.to_numeric(df[lower_map["mid"]], errors="coerce")
    elif {"bid", "ask"}.issubset(lower_map.keys()):
        bid = pd.to_numeric(df[lower_map["bid"]], errors="coerce")
        ask = pd.to_numeric(df[lower_map["ask"]], errors="coerce")
        mid = (bid + ask) / 2.0
    else:
        for k in ["price", "close", "last", "c", "close price"]:
            if k in lower_map:
                mid = pd.to_numeric(df[lower_map[k]], errors="coerce")
                break
        if mid is None:
            df2 = _read_csv_any_sep(path, header=None)
            df2.columns = [f"col{i}" for i in range(df2.shape[1])]
            ts_text = df2["col0"].astype(str)
            if df2.shape[1] >= 3 and _series_is_numeric(df2["col1"]) and _series_is_numeric(df2["col2"]):
                mid = (pd.to_numeric(df2["col1"], errors="coerce") + pd.to_numeric(df2["col2"], errors="coerce")) / 2.0
            elif df2.shape[1] >= 2 and _series_is_numeric(df2["col1"]):
                mid = pd.to_numeric(df2["col1"], errors="coerce")
            else:
                raise ValueError(f"{path} missing usable price columns. Headers: {list(df.columns)}")

    return _finalize(ts_text, mid)

def load_ticks(glob_pattern: str) -> pd.DataFrame:
    files = sorted(Path().glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No tick CSVs found under {glob_pattern}")
    frames = [read_tick_csv(p) for p in files]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[np.isfinite(df["mid"])]
    return df


# ============================== Bars & Windows ==============================

@dataclass
class WindowDef:
    key: str
    start: str
    end: str

def _to_pt_naive(utc_index: pd.DatetimeIndex, tz_pt: str) -> pd.Series:
    pt = utc_index.tz_convert(tz_pt)
    return pd.Series(pt.tz_localize(None), index=utc_index)

def _parse_hhmm(hhmm: str) -> Tuple[int, int]:
    h, m = hhmm.split(":")
    return int(h), int(m)

def build_session_bars_for_window(ticks_utc: pd.DataFrame, win: WindowDef, tz_pt: str) -> pd.DataFrame:
    df = ticks_utc.copy()
    pt_naive = _to_pt_naive(df.index, tz_pt)          # naive PT
    df = df.assign(pt_dt=pt_naive.values)
    df["pt_date"] = df["pt_dt"].dt.date

    from datetime import time as dtime
    sh, sm = _parse_hhmm(win.start); eh, em = _parse_hhmm(win.end)
    start_t = dtime(sh, sm); end_t = dtime(eh, em)

    def in_window(s: pd.Series) -> pd.Series:
        t = s.dt.time
        return (t >= start_t) & (t <= end_t) if start_t <= end_t else ((t >= start_t) | (t <= end_t))

    groups = []
    for d, sub in df.groupby("pt_date", sort=True):
        mask = in_window(sub["pt_dt"])
        s = sub.loc[mask]
        if s.empty:
            continue

        o = s["mid"].iloc[0]; h = s["mid"].max(); l = s["mid"].min(); c = s["mid"].iloc[-1]

        # --- FIX 1: localize with explicit DST rules (handles 2023-11-05 01:00 twice) ---
        pt_index = pd.DatetimeIndex(s["pt_dt"])
        try:
            pt_tz = pt_index.tz_localize(tz_pt, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            # fallback if "infer" can’t decide: assume DST=True for duplicates
            pt_tz = pt_index.tz_localize(tz_pt, ambiguous=True, nonexistent="shift_forward")

        spt = s.set_index(pt_tz)

        # --- FIX 2: use 'min' instead of '1T' (FutureWarning) ---
        m1 = spt["mid"].resample("min").last().ffill()

        rv = (np.log(m1).diff() ** 2).sum()
        groups.append({
            "pt_date": pd.to_datetime(str(d)),
            "open": o, "high": h, "low": l, "close": c,
            "rv": rv, "tick_count": len(s)
        })

    out = pd.DataFrame(groups).set_index("pt_date").sort_index()
    out.columns = pd.MultiIndex.from_product([[win.key], out.columns])
    return out


def build_all_bars(ticks_utc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    tz_pt = cfg["pt_timezone"]
    wins = [
        WindowDef("daily", cfg["windows"]["daily"]["start"], cfg["windows"]["daily"]["end"]),
        WindowDef("w2_10", cfg["windows"]["w2_10"]["start"], cfg["windows"]["w2_10"]["end"]),
        WindowDef("w3_12", cfg["windows"]["w3_12"]["start"], cfg["windows"]["w3_12"]["end"]),
    ]
    bars = [build_session_bars_for_window(ticks_utc, w, tz_pt) for w in wins]
    out = pd.concat(bars, axis=1).sort_index()
    return out


# ============================== Features ====================================

def true_range(h: pd.Series, l: pd.Series, c_prev: pd.Series) -> pd.Series:
    v1 = h - l
    v2 = (h - c_prev).abs()
    v3 = (l - c_prev).abs()
    return pd.concat([v1, v2, v3], axis=1).max(axis=1)

def atr(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    tr = true_range(h, l, c.shift(1))
    return tr.rolling(n, min_periods=max(2, n // 3)).mean()

def round_level_features(close: pd.Series, cfg: dict) -> pd.DataFrame:
    pip = cfg["pip_size"]
    steps = {10: 0.0010, 25: 0.0025, 50: 0.0050, 100: 0.0100}
    cols = {}
    for k in cfg["round_levels_pips"]:
        k = int(k)
        step = steps[k]
        level = (close / step).round() * step
        cols[f"dist_L{k}_pips"] = (close - level).abs() / pip
        cols[f"near_L{k}"] = (cols[f"dist_L{k}_pips"] <= cfg["round_proximity_thresh_pips"]).astype(int)
    return pd.DataFrame(cols, index=close.index)

def breakout_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    rng_n: int,
    atr_n: int,
    gamma: float,
    fail_backoff_pips: int,
    pip_size: float,
) -> pd.DataFrame:
    range_high = high.rolling(rng_n, min_periods=2).max().shift(1)
    range_low = low.rolling(rng_n, min_periods=2).min().shift(1)
    _atr = atr(None, high, low, close, atr_n)

    up_th = range_high + gamma * _atr
    dn_th = range_low - gamma * _atr

    break_up = (high > up_th).astype(int)
    break_dn = (low < dn_th).astype(int)

    hold_up = (close > range_high).astype(int)
    hold_dn = (close < range_low).astype(int)

    false_up = ((break_up == 1) & (hold_up == 0) & (close < (range_high - fail_backoff_pips * pip_size))).astype(int)
    false_dn = ((break_dn == 1) & (hold_dn == 0) & (close > (range_low + fail_backoff_pips * pip_size))).astype(int)

    ext_up_pips = (high - range_high) / pip_size
    ext_dn_pips = (range_low - low) / pip_size

    return pd.DataFrame(
        {
            "break_up": break_up,
            "break_dn": break_dn,
            "false_up": false_up,
            "false_dn": false_dn,
            "break_ext_up_pips": ext_up_pips.clip(lower=0),
            "break_ext_dn_pips": ext_dn_pips.clip(lower=0),
            "atr": _atr,
        },
        index=close.index,
    )

def exhaustion_features(close: pd.Series, high: pd.Series, low: pd.Series, o: pd.Series, cfg: dict) -> pd.DataFrame:
    n = cfg["exhaustion"]["ma_window"]
    atr_n = cfg["exhaustion"]["atr_window"]
    pip = cfg["pip_size"]

    ma = close.rolling(n, min_periods=max(2, n // 3)).mean()
    std = close.rolling(n, min_periods=max(2, n // 3)).std()
    z_ma = (close - ma) / (std + 1e-9)
    _atr = atr(o, high, low, close, atr_n)
    atr_gap = (close - ma) / (_atr + 1e-9)

    bb_upper = ma + 2 * std
    bb_lower = ma - 2 * std
    bb_pct = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

    upper_wick = high - pd.concat([o, close], axis=1).max(axis=1)
    lower_wick = pd.concat([o, close], axis=1).min(axis=1) - low

    k = 3
    mom = close - close.shift(k)
    ema5 = mom.ewm(span=5, adjust=False).mean()
    mom_slope = ema5.diff()

    return pd.DataFrame(
        {
            "z_ma": z_ma,
            "atr_gap": atr_gap,
            "bb_pct": bb_pct,
            "upper_wick_pips": upper_wick / pip,
            "lower_wick_pips": lower_wick / pip,
            "mom_k3": mom / pip,
            "mom_slope": mom_slope / pip,
        },
        index=close.index,
    )

def session_flags(index: pd.DatetimeIndex, window_key: str) -> pd.DataFrame:
    dow = pd.get_dummies(pd.Series(index.dayofweek, index=index).rename("dow"), prefix="dow")
    eom = (index.to_period("M").to_timestamp("M") == index).astype(int)
    q_end = (index.to_period("Q").to_timestamp("Q") == index).astype(int)
    flags = pd.DataFrame(dow, index=index)
    flags["is_eom"] = eom
    flags["is_eoq"] = q_end
    flags[f"is_{window_key}"] = 1
    return flags

def tick_volume_features(
    ticks_utc: pd.DataFrame, window_bars: pd.DataFrame, window_key: str, tz_pt: str, baseline_days: int
) -> pd.DataFrame:
    df = ticks_utc.copy()
    pt_ts = _to_pt_naive(df.index, tz_pt)
    df = df.assign(pt_dt=pt_ts.values)
    df["pt_date"] = df["pt_dt"].dt.date

    days = window_bars.index
    per_day_profiles = {}
    for d in days:
        day_ticks = df[df["pt_date"] == d.date()]
        if day_ticks.empty:
            continue
        spt = day_ticks.set_index(pd.DatetimeIndex(day_ticks["pt_dt"]).tz_localize(tz_pt))
        m1 = spt["mid"].resample("1T").count()
        per_day_profiles[d] = m1

    if not per_day_profiles:
        return pd.DataFrame(index=window_bars.index)

    aligned = {d: s.values.astype(float) for d, s in per_day_profiles.items()}
    max_len = max(len(v) for v in aligned.values())
    M = pd.DataFrame({d: np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for d, v in aligned.items()}).T
    M.index = pd.to_datetime(M.index)

    baseline = M.rolling(baseline_days, min_periods=5).median()
    intensity = M / (baseline + 1e-9)
    tick_rate_mean = M.mean(axis=1)
    tick_rate_max = M.max(axis=1)
    accel_fast = intensity.ewm(span=5, adjust=False).mean()
    accel_slow = intensity.ewm(span=20, adjust=False).mean()
    accel = (accel_fast - accel_slow).max(axis=1)
    pct_hot = (intensity > 1.5).mean(axis=1)

    feats = pd.DataFrame(
        {
            f"{window_key}_tick_rate_mean": tick_rate_mean,
            f"{window_key}_tick_rate_max": tick_rate_max,
            f"{window_key}_tick_accel_max": accel,
            f"{window_key}_pct_time_hot": pct_hot,
        }
    )
    feats.index.name = "pt_date"
    feats = feats.reindex(window_bars.index).fillna(method="ffill").fillna(0.0)
    return feats

def microstructure_imbalance(ticks_utc: pd.DataFrame, window_bars: pd.DataFrame, tz_pt: str, window_key: str) -> pd.Series:
    df = ticks_utc.copy()
    pt_naive = _to_pt_naive(df.index, tz_pt)
    df = df.assign(pt_dt=pt_naive.values)
    df["pt_date"] = df["pt_dt"].dt.date

    imbs = []
    for d in window_bars.index:
        day_ticks = df[df["pt_date"] == d.date()]
        if day_ticks.empty:
            imbs.append(np.nan)
            continue
        mid = day_ticks["mid"].to_numpy()
        diff = np.sign(np.diff(mid))
        upticks = (diff > 0).sum()
        downticks = (diff < 0).sum()
        imb = (upticks - downticks) / (upticks + downticks + 1e-9)
        imbs.append(imb)
    return pd.Series(imbs, index=window_bars.index, name=f"{window_key}_ud_imbalance")

def pre_news_features(calendar_csv: str, window_bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    path = Path(calendar_csv)
    if not path.exists():
        return pd.DataFrame(index=window_bars.index)
    cal = pd.read_csv(path)
    need = {"event_time_utc", "impact", "currency"}
    low = set(c.lower() for c in cal.columns)
    if not need.issubset(low):
        return pd.DataFrame(index=window_bars.index)
    cal.columns = [c.lower() for c in cal.columns]
    cal["event_time_utc"] = pd.to_datetime(cal["event_time_utc"], utc=True, errors="coerce")
    cal = cal[cal["impact"].str.upper().isin([i.upper() for i in cfg["pre_news"]["impacts"]])]
    cal = cal[cal["currency"].str.upper().isin(cfg["pre_news"]["currencies"])].copy()
    if cal.empty:
        return pd.DataFrame(index=window_bars.index)

    tz_pt = cfg["pt_timezone"]
    mins_to_next = []
    had_recent = []
    pre_window = cfg["pre_news"]["pre_window_minutes"]
    post_window = cfg["pre_news"]["post_window_minutes"]
    for d in window_bars.index:
        pt_open = pd.Timestamp(d.date()).tz_localize(tz_pt).tz_convert("UTC")
        future = cal[cal["event_time_utc"] >= pt_open]
        past = cal[(cal["event_time_utc"] < pt_open) & (cal["event_time_utc"] >= pt_open - pd.Timedelta(minutes=post_window))]
        if future.empty:
            mins_to_next.append(np.nan)
        else:
            dt = (future["event_time_utc"].min() - pt_open).total_seconds() / 60.0
            mins_to_next.append(dt)
        had_recent.append(1 if not past.empty else 0)

    s = pd.Series(mins_to_next, index=window_bars.index)
    feats = pd.DataFrame(
        {
            "minutes_to_next_high_impact": s.clip(lower=0).fillna(1440).clip(0, 1440),
            "is_post_news": pd.Series(had_recent, index=window_bars.index),
            "is_pre_news": s.between(0, pre_window).astype(int).fillna(0),
        }
    )
    return feats

def assemble_features(bars: pd.DataFrame, ticks_utc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    tz_pt = cfg["pt_timezone"]
    pip = cfg["pip_size"]
    blocks = []

    for wkey in ["daily", "w2_10", "w3_12"]:
        o = bars[(wkey, "open")]; h = bars[(wkey, "high")]
        l = bars[(wkey, "low")];  c = bars[(wkey, "close")]
        blk = []
        blk.append(round_level_features(c, cfg).add_prefix(f"{wkey}_"))
        blk.append(breakout_features(h, l, c, cfg["breakout"]["lookback_sessions"],
                                     cfg["breakout"]["atr_window"], cfg["breakout"]["breakout_gamma"],
                                     cfg["breakout"]["fail_backoff_pips"], pip).add_prefix(f"{wkey}_"))
        blk.append(exhaustion_features(c, h, l, o, cfg).add_prefix(f"{wkey}_"))
        blk.append(session_flags(c.index, wkey))
        blk.append(tick_volume_features(ticks_utc, bars[wkey], wkey, tz_pt, cfg["tick_volume"]["baseline_days"]))
        blk.append(microstructure_imbalance(ticks_utc, bars[wkey], tz_pt, wkey).to_frame())
        blk.append(pre_news_features(cfg["paths"]["econ_calendar"], bars[wkey], cfg).add_prefix(f"{wkey}_"))
        feats = pd.concat(blk, axis=1)
        feats["window"] = wkey
        blocks.append(feats)

    features = pd.concat(blocks, axis=0).sort_index()
    return features


# ============================== Labels & Dataset ============================

def compute_labels(bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pip = cfg["pip_size"]
    out = []
    for wkey in ["daily", "w2_10", "w3_12"]:
        o = bars[(wkey, "open")]; c = bars[(wkey, "close")]
        pips_delta = (c - o) / pip
        up = (pips_delta > 0).astype(int)
        df = pd.DataFrame({f"{wkey}_pips": pips_delta, f"{wkey}_up": up}, index=o.index)
        df["window"] = wkey
        out.append(df)
    return pd.concat(out, axis=0).sort_index()

def build_dataset(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for wkey in ["daily", "w2_10", "w3_12"]:
        f = features[features["window"] == wkey].copy()
        if f.empty:
            continue
        l = labels[labels["window"] == wkey].copy().drop(columns=["window"], errors="ignore")
        d = f.join(l, how="inner")
        parts.append(d)
    return pd.concat(parts, axis=0).sort_index() if parts else pd.DataFrame()


# ============================== CV & Models =================================

def time_splits_with_embargo(dates: pd.DatetimeIndex, n_splits: int, embargo_days: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    uniq_days = pd.Index(sorted(dates.unique()))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for tr_idx, te_idx in tscv.split(uniq_days):
        train_days = uniq_days[tr_idx]
        test_days = uniq_days[te_idx]
        if embargo_days > 0:
            cutoff = test_days.min() - pd.Timedelta(days=embargo_days)
            train_days = train_days[train_days <= cutoff]
        train_mask = dates.isin(train_days)
        test_mask = dates.isin(test_days)
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return splits

@dataclass
class Models:
    cls: Pipeline
    reg: Pipeline

def build_models() -> Models:
    cls = Pipeline([("scaler", StandardScaler(with_mean=True)),
                    ("logit", LogisticRegression(max_iter=2000, solver="lbfgs"))])
    reg = Pipeline([("scaler", StandardScaler(with_mean=True)),
                    ("huber", HuberRegressor(max_iter=2000, epsilon=1.35, alpha=1e-4))])
    return Models(cls=cls, reg=reg)


# ============================== Manifests ===================================

def _save_cols(models_dir: Path, wkey: str, cols: List[str]) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"cols_{wkey}.json").write_text(json.dumps(cols), encoding="utf-8")

def _load_cols(models_dir: Path, wkey: str) -> List[str]:
    p = models_dir / f"cols_{wkey}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing column manifest: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def _synthesize_cols_from_dataset(df_cols, wkey: str) -> list:
    cols = (
        [c for c in df_cols if c.startswith(f"{wkey}_")]
        + [c for c in df_cols if c.startswith("dow_")]
        + ["is_eom", "is_eoq", f"is_{wkey}"]
    )
    seen = set()
    return [c for c in cols if not (c in seen or seen.add(c))]

def _cols_for_window_like_training(sub: pd.DataFrame, wkey: str, models_dir: Path) -> pd.DataFrame:
    """Return X with exact train-time column order, adding missing cols as 0.0."""
    try:
        cols = _load_cols(models_dir, wkey)
    except FileNotFoundError:
        cols = _synthesize_cols_from_dataset(sub.columns, wkey)
        _save_cols(models_dir, wkey, cols)
    X = sub.copy()
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


# ============================== Train / Eval ================================

def train_and_eval_per_window(df: pd.DataFrame, cfg: dict, models_dir: Path) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}

    for wkey in ["daily", "w2_10", "w3_12"]:
        sub = df[df["window"] == wkey].copy()
        if sub.empty:
            print(f"[WARN] No rows for window {wkey}")
            continue

        sub.replace([np.inf, -np.inf], np.nan, inplace=True)
        sub.dropna(subset=[f"{wkey}_up", f"{wkey}_pips"], inplace=True)
        if sub.empty:
            print(f"[WARN] No usable rows after dropping NaNs for {wkey}")
            continue

        y_cls_all = sub[f"{wkey}_up"].astype(int)
        y_reg_all = sub[f"{wkey}_pips"].astype(float)

        feat_cols = (
            [c for c in sub.columns if c.startswith(f"{wkey}_") and not (c.endswith("_up") or c.endswith("_pips"))]
            + [c for c in sub.columns if c.startswith("dow_")]
            + ["is_eom", "is_eoq", f"is_{wkey}"]
        )
        seen = set()
        feat_cols = [c for c in feat_cols if not (c in seen or seen.add(c))]
        (Path(models_dir) / f"cols_{wkey}.json").write_text(json.dumps(feat_cols), encoding="utf-8")

        X_all = sub[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        splits = time_splits_with_embargo(sub.index, cfg["cv"]["n_splits"], cfg["cv"]["embargo_days"])
        aucs, briers, accs, maes = [], [], [], []

        best_cls = best_reg = None
        best_score = -np.inf

        for (tr_idx, te_idx) in splits:
            if len(tr_idx) == 0 or len(te_idx) == 0:
                continue

            Xtr, Xte = X_all.iloc[tr_idx], X_all.iloc[te_idx]
            ytr_cls, yte_cls = y_cls_all.iloc[tr_idx], y_cls_all.iloc[te_idx]
            ytr_reg, yte_reg = y_reg_all.iloc[tr_idx], y_reg_all.iloc[te_idx]

            tr_keep = ytr_reg.notna()
            te_keep = yte_reg.notna()
            if tr_keep.sum() == 0 or te_keep.sum() == 0:
                continue

            Xtr, ytr_cls, ytr_reg = Xtr.loc[tr_keep], ytr_cls.loc[tr_keep], ytr_reg.loc[tr_keep]
            Xte, yte_cls, yte_reg = Xte.loc[te_keep], yte_cls.loc[te_keep], yte_reg.loc[te_keep]

            m = build_models()
            m.cls.fit(Xtr, ytr_cls)
            m.reg.fit(Xtr, ytr_reg)

            proba_up = m.cls.predict_proba(Xte)[:, 1]
            pred_up = (proba_up >= 0.5).astype(int)
            auc = roc_auc_score(yte_cls, proba_up) if len(np.unique(yte_cls)) > 1 else np.nan
            brier = brier_score_loss(yte_cls, proba_up)
            acc = accuracy_score(yte_cls, pred_up)
            mae = mean_absolute_error(yte_reg, m.reg.predict(Xte))

            aucs.append(auc); briers.append(brier); accs.append(acc); maes.append(mae)
            score = np.nan_to_num(auc, nan=0.0) + acc * 0.1 - mae * 0.001
            if score > best_score:
                best_score, best_cls, best_reg = score, m.cls, m.reg

        if best_cls is None or best_reg is None:
            print(f"[WARN] No valid CV folds for {wkey}; skipping persist.")
            continue

        dump(best_cls, Path(models_dir) / f"model_cls_{wkey}.joblib")
        dump(best_reg, Path(models_dir) / f"model_reg_{wkey}.joblib")

        metrics[wkey] = {
            "AUC_mean": float(np.nanmean(aucs)),
            "Brier_mean": float(np.mean(briers)),
            "Acc_mean": float(np.mean(accs)),
            "MAE_pips_mean": float(np.mean(maes)),
            "n_samples": int(len(sub)),
        }

    return metrics


# ========================= Threshold tuning & policy ========================

def _oof_probs_for_window(df: pd.DataFrame, cfg: dict, wkey: str, models_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df[df["window"] == wkey].copy()
    if sub.empty:
        return np.array([]), np.array([]), np.array([])

    sub.replace([np.inf, -np.inf], np.nan, inplace=True)
    sub.dropna(subset=[f"{wkey}_up", f"{wkey}_pips"], inplace=True)
    if sub.empty:
        return np.array([]), np.array([]), np.array([])

    y_cls = sub[f"{wkey}_up"].astype(int).to_numpy()
    y_pips = sub[f"{wkey}_pips"].astype(float).to_numpy()
    X = _cols_for_window_like_training(sub, wkey, models_dir)

    splits = time_splits_with_embargo(sub.index, cfg["cv"]["n_splits"], cfg["cv"]["embargo_days"])
    proba = np.full(len(sub), np.nan, dtype=float)

    for tr_idx, te_idx in splits:
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr_cls = y_cls[tr_idx]
        m = build_models()
        m.cls.fit(Xtr, ytr_cls)
        proba[te_idx] = m.cls.predict_proba(Xte)[:, 1]

    mask = ~np.isnan(proba)
    return proba[mask], y_cls[mask], y_pips[mask]

def _tune_thresholds(y: np.ndarray, p: np.ndarray, true_pips: np.ndarray) -> Dict[str, Dict[str, float]]:
    if len(y) == 0:
        return {"auc": float("nan"), "flip": False, "acc": {}, "mcc": {}, "pnl": {}}

    auc_val = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    grid = np.linspace(0.35, 0.65, 61)
    best = {"acc": (-1.0, 0.5), "mcc": (-1.0, 0.5), "pnl": (-1e99, 0.5)}

    for t in grid:
        pred = (p >= t).astype(int)
        acc = float((pred == y).mean())
        if acc > best["acc"][0]:
            best["acc"] = (acc, float(t))
        mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else -1.0
        if mcc > best["mcc"][0]:
            best["mcc"] = (float(mcc), float(t))
        dir_pred = np.where(pred == 1, 1.0, -1.0)
        pnl = float(np.sum(dir_pred * true_pips))
        if pnl > best["pnl"][0]:
            best["pnl"] = (pnl, float(t))

    flip_bool = bool(np.isfinite(auc_val) and (auc_val < 0.5))

    return {
        "auc": auc_val,
        "flip": flip_bool,
        "acc": {"score": float(best["acc"][0]), "threshold": float(best["acc"][1])},
        "mcc": {"score": float(best["mcc"][0]), "threshold": float(best["mcc"][1])},
        "pnl": {"score": float(best["pnl"][0]), "threshold": float(best["pnl"][1])},
    }

def _to_native_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_native_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

def build_and_save_policies(df: pd.DataFrame, cfg: dict, models_dir: Path) -> Dict[str, dict]:
    models_dir = ensure_dir(models_dir)
    policies = {}
    for wkey in ["daily", "w2_10", "w3_12"]:
        p, y, pips = _oof_probs_for_window(df, cfg, wkey, models_dir)
        pol = _tune_thresholds(y, p, pips)
        auc = pol["auc"]
        pol["active"] = bool(np.isfinite(auc) and auc >= 0.55)
        if wkey == "w2_10" and (not pol["active"]):
            pol["note"] = "Window below AUC guard; keep inactive or use flip cautiously."
        path = Path(models_dir) / f"policy_{wkey}.json"
        path.write_text(json.dumps(_to_native_jsonable(pol), indent=2), encoding="utf-8")
        policies[wkey] = pol
    return policies

def _load_policy(models_dir: Path, wkey: str) -> dict:
    path = Path(models_dir) / f"policy_{wkey}.json"
    if not path.exists():
        return {"flip": False, "active": True,
                "acc": {"threshold": 0.5}, "mcc": {"threshold": 0.5}, "pnl": {"threshold": 0.5}}
    return json.loads(path.read_text(encoding="utf-8"))


# ============================== Predict =====================================

def _get_feat_matrix_for_predict(sub: pd.DataFrame, wkey: str, models_dir: Path) -> pd.DataFrame:
    """Load train-time feature list; if missing, derive from dataset and save it."""
    try:
        feat_cols = _load_cols(models_dir, wkey)
    except FileNotFoundError:
        feat_cols = _synthesize_cols_from_dataset(sub.columns, wkey)
        _save_cols(models_dir, wkey, feat_cols)

    X = sub.copy()
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X

def predict_next_day(features: pd.DataFrame, cfg: dict, models_dir: Path, objective: str = "acc") -> Dict[str, Dict[str, float]]:
    out = {}
    for wkey in ["daily", "w2_10", "w3_12"]:
        sub = features[features["window"] == wkey].copy()
        if sub.empty:
            continue

        X = _get_feat_matrix_for_predict(sub, wkey, models_dir)
        x_last = X.iloc[[-1]]

        cls = load(Path(models_dir) / f"model_cls_{wkey}.joblib")
        reg = load(Path(models_dir) / f"model_reg_{wkey}.joblib")

        proba_raw = float(cls.predict_proba(x_last)[:, 1][0])

        pol = _load_policy(models_dir, wkey)
        flip = bool(pol.get("flip", False))
        thr = float(pol.get(objective, {}).get("threshold", 0.5))
        proba_eff = 1.0 - proba_raw if flip else proba_raw
        pred_dir = "UP" if proba_eff >= thr else "DOWN"
        pips_pred = float(reg.predict(x_last)[0])

        key_out = wkey if wkey != "w2_10" else "2_10"
        out[key_out] = {
            "dir": pred_dir,
            "proba_raw": round(proba_raw, 4),
            "proba_eff": round(proba_eff, 4),
            "threshold": round(thr, 4),
            "flip": flip,
            "pips_pred": round(pips_pred, 2),
            "active": bool(pol.get("active", True)),
        }
    return out


# ============================== Build Orchestration =========================

def build_all(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ticks = load_ticks(cfg["paths"]["ticks_glob"])
    bars = build_all_bars(ticks, cfg)
    feats = assemble_features(bars, ticks, cfg)
    labels = compute_labels(bars, cfg)
    dataset = build_dataset(feats, labels)

    outp = Path(cfg["paths"]["features_out"])
    outp.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(outp)
    return ticks, bars, dataset

def rebuild_from_checkpoints(cfg: dict) -> Optional[pd.DataFrame]:
    feats_p = Path(cfg["paths"]["feats_only"])
    labs_p = Path(cfg["paths"]["labs_only"])
    if feats_p.exists() and labs_p.exists():
        F = pd.read_parquet(feats_p)
        L = pd.read_parquet(labs_p)
        mins_cols = [c for c in F.columns if c.endswith("minutes_to_next_high_impact")]
        for c in mins_cols:
            F[c] = pd.to_numeric(F[c], errors="coerce").fillna(1440).clip(0, 1440)
        dataset = build_dataset(F, L)
        outp = Path(cfg["paths"]["features_out"])
        outp.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(outp)
        return dataset
    return None


# ============================== Holdout Eval ================================

# --- replace the existing evaluate_holdout with this version (NO PERSIST) ---
def evaluate_holdout(df: pd.DataFrame, cfg: dict, cutoff: pd.Timestamp) -> Dict[str, Dict[str, float]]:
    """
    Forward holdout that does NOT write models or manifests to disk.
    Trains per-window models on df[df.index < cutoff] and evaluates on df[df.index >= cutoff].
    Returns metrics per window.
    """
    res: Dict[str, Dict[str, float]] = {}

    train_df = df[df.index < cutoff].copy()
    test_df  = df[df.index >= cutoff].copy()
    if train_df.empty or test_df.empty:
        return res

    def _feat_cols_from_training(cols, wkey: str) -> list:
        # Synthesize the exact training-time feature list WITHOUT saving anything.
        base = (
            [c for c in cols if c.startswith(f"{wkey}_") and not (c.endswith("_up") or c.endswith("_pips"))]
            + [c for c in cols if c.startswith("dow_")]
            + ["is_eom", "is_eoq", f"is_{wkey}"]
        )
        seen = set()
        return [c for c in base if not (c in seen or seen.add(c))]

    for wkey in ["daily", "w2_10", "w3_12"]:
        tr = train_df[train_df["window"] == wkey].copy()
        te = test_df[test_df["window"] == wkey].copy()
        if tr.empty or te.empty:
            continue

        # Targets
        ytr_cls = tr[f"{wkey}_up"].astype(int)
        ytr_reg = tr[f"{wkey}_pips"].astype(float)
        yte_cls = te[f"{wkey}_up"].astype(int)
        yte_reg = te[f"{wkey}_pips"].astype(float)

        # Feature columns from TRAINING ONLY (no disk I/O)
        feat_cols = _feat_cols_from_training(tr.columns, wkey)

        def _prep_X(df_sub: pd.DataFrame, cols: list) -> pd.DataFrame:
            X = df_sub.copy()
            for c in cols:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return X

        Xtr = _prep_X(tr, feat_cols)
        Xte = _prep_X(te, feat_cols)

        # Fit models in-memory
        m = build_models()
        # Drop rows with NaNs in targets for safety
        keep_tr = ytr_reg.notna()
        keep_te = yte_reg.notna()
        if keep_tr.sum() == 0 or Xtr.empty:
            continue

        m.cls.fit(Xtr.loc[keep_tr], ytr_cls.loc[keep_tr])
        m.reg.fit(Xtr.loc[keep_tr], ytr_reg.loc[keep_tr])

        # Classification metrics
        auc = float("nan"); brier = float("nan"); acc = float("nan")
        if yte_cls.notna().any():
            proba_up = m.cls.predict_proba(Xte)[:, 1]
            y_true   = yte_cls.to_numpy()
            if len(np.unique(y_true)) > 1:
                auc = float(roc_auc_score(y_true, proba_up))
            brier = float(brier_score_loss(y_true, proba_up))
            acc   = float(accuracy_score(y_true, (proba_up >= 0.5).astype(int)))

        # Regression metric
        mae = float("nan")
        if keep_te.any():
            y_pred = m.reg.predict(Xte.loc[keep_te])
            mae = float(mean_absolute_error(yte_reg.loc[keep_te], y_pred))

        res[wkey] = {
            "AUC": auc,
            "Brier": brier,
            "Acc": acc,
            "MAE_pips": mae,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "cutoff": str(pd.Timestamp(cutoff).date()),
        }

    return res



# ============================== CLI =========================================

def main():
    ap = argparse.ArgumentParser(description="FX pipeline (single-file)")
    ap.add_argument("--config", type=str, default=None, help="Optional YAML config")
    ap.add_argument("--ticks-glob", type=str, default=None, help="Override ticks glob")
    ap.add_argument("--symbol", type=str, default=None, help="Limit to a single symbol (e.g., EURUSD)")

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="Ingest ticks→bars→features+labels→dataset parquet")
    sub.add_parser("train", help="Train per-window models, persist manifests + policies")

    ap_pred = sub.add_parser("predict", help="Predict next session/day from latest dataset")
    ap_pred.add_argument(
        "--objective",
        choices=["acc", "mcc", "pnl"],
        default="acc",
        help="Decision rule to use (default: acc)",
    )
    ap_pred.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Optional: per-symbol run; switches features/models to symbol-specific paths (e.g., EURUSD, USDCHF)",
    )


    hp = sub.add_parser("holdout", help="Forward holdout: train < cutoff, test ≥ cutoff")
    g = hp.add_mutually_exclusive_group(required=True)
    g.add_argument("--cutoff", type=str, help="Cutoff date YYYY-MM-DD")
    g.add_argument("--last-days", type=int, help="Use last N days as holdout (compute cutoff)")

    # ---- parse & config ----
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config)
    if args.ticks_glob:
        cfg["paths"]["ticks_glob"] = args.ticks_glob

    # ---- per-symbol overrides (namespacing artifacts) ----
    sym = (args.symbol or "").upper().replace("/", "") if getattr(args, "symbol", None) else ""
    if sym:
        cfg["paths"]["ticks_glob"]   = f"data/ticks/*{sym}*.csv"
        cfg["paths"]["features_out"] = f"data/features_{sym}.parquet"
        cfg["paths"]["feats_only"]   = f"data/features_only_{sym}.parquet"
        cfg["paths"]["labs_only"]    = f"data/labels_only_{sym}.parquet"
        cfg["paths"]["models_dir"]   = f"models/{sym}"

    if getattr(args, "symbol", None):
            sym = (args.symbol or "").upper().replace("/", "")
            if sym:
                cfg["paths"]["ticks_glob"]   = f"data/ticks/*{sym}*.csv"
                cfg["paths"]["features_out"] = f"data/features_{sym}.parquet"
                cfg["paths"]["feats_only"]   = f"data/features_only_{sym}.parquet"
                cfg["paths"]["labs_only"]    = f"data/labels_only_{sym}.parquet"
                cfg["paths"]["models_dir"]   = f"models/{sym}"
       
    # IMPORTANT: derive these AFTER overrides
    models_dir = ensure_dir(cfg["paths"]["models_dir"])
    features_p = Path(cfg["paths"]["features_out"])

    # ---- commands ----
    if args.cmd == "build":
        ds = rebuild_from_checkpoints(cfg)
        if ds is not None:
            print(f"[build] salvaged -> {features_p} with {len(ds)} rows")
            return
        _, _, ds = build_all(cfg)
        print(f"[build] wrote {features_p} with {len(ds)} rows")

    elif args.cmd == "train":
        if not features_p.exists():
            ds = rebuild_from_checkpoints(cfg)
            if ds is None:
                print("[train] features parquet missing. Running build.")
                _, _, ds = build_all(cfg)
        df = pd.read_parquet(features_p)
        ensure_manifests_from_dataset(df, models_dir)
        metrics = train_and_eval_per_window(df, cfg, models_dir)
        print(json.dumps(metrics, indent=2))
        policies = build_and_save_policies(df, cfg, models_dir)
        print("[train] saved per-window policies:")
        print(json.dumps(policies, indent=2))

    elif args.cmd == "predict":
        if not features_p.exists():
            ds = rebuild_from_checkpoints(cfg)
            if ds is None:
                print("[predict] features parquet missing. Running build+train first.")
                _, _, _ = build_all(cfg)
                df2 = pd.read_parquet(features_p)
                _ = train_and_eval_per_window(df2, cfg, models_dir)
                _ = build_and_save_policies(df2, cfg, models_dir)
        df = pd.read_parquet(features_p)
        feats_for_pred = df.drop(columns=[c for c in df.columns if c.endswith("_pips") or c.endswith("_up")]).copy()
        preds = predict_next_day(feats_for_pred, cfg, models_dir, objective=getattr(args, "objective", "acc"))
        print(json.dumps(preds, indent=2))

    elif args.cmd == "holdout":
        if not features_p.exists():
            ds = rebuild_from_checkpoints(cfg)
            if ds is None:
                print("[holdout] features parquet missing. Running build.")
                _, _, _ = build_all(cfg)
        df = pd.read_parquet(features_p)
        if getattr(args, "cutoff", None):
            cutoff = pd.Timestamp(args.cutoff)
        else:
            last_days = int(args.last_days)
            cutoff = df.index.max() - pd.Timedelta(days=last_days)
        res = evaluate_holdout(df, cfg, cutoff)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
