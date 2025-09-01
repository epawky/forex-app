# -*- coding: utf-8 -*-
# fx_pipeline.py
# Single-file EURUSD pipeline: robust headless CSV ingest, per-window features,
# manifest-locked training/prediction, threshold policies (acc/mcc/pnl),
# and forward-holdout evaluation. Python 3.8+

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================== Default Config ==============================

DEFAULT_CONFIG = {
    "pip_size": 0.0001,
    "pt_timezone": "America/Los_Angeles",
    "paths": {
        "ticks_glob": "data/ticks/*.csv",
        "econ_calendar": "data/econ_calendar.csv",
        "features_out": "data/features.parquet",
        "models_dir": "models",
        "feats_only": "data/features_only.parquet",  # optional salvage
        "labs_only": "data/labels_only.parquet",     # optional salvage
    },
    "windows": {
        "daily": {"start": "00:00", "end": "23:59"},
        "w2_10": {"start": "02:00", "end": "10:00"},  # ET-like example; adjust if needed
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
    "tick_volume": {
        "baseline_days": 20,
        "hot_threshold": 1.5,
    },
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
            import yaml
            with p.open("r", encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
            # shallow merge at top-level + dict children
            for k, v in user.items():
                if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
    return cfg

def ensure_dir(d: str | Path) -> Path:
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_manifests_from_dataset(df: pd.DataFrame, models_dir: Path) -> None:
    """
    Create per-window column manifests (cols_*.json) if they don't exist yet,
    using the current dataset columns. This keeps train/predict aligned.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    for wkey in ["daily", "w2_10", "w3_12"]:
        p = models_dir / f"cols_{wkey}.json"
        if p.exists():
            continue
        if f"is_{wkey}" not in df.columns:
            # dataset doesn't contain this window; skip
            continue

        feat_cols = (
            [c for c in df.columns
             if c.startswith(f"{wkey}_") and not (c.endswith("_up") or c.endswith("_pips"))]
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
        "timestamp","datetime","date_time","date time","gmt time","utc time",
        "time gmt","time (gmt)","time (utc)","local time","bar time","bar start time","time","date"
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
    Supports headered and headless CSVs. Headless:
      col0=full datetime, col1=price  OR  col0=date, col1=time, col2=bid, col3=ask
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

def build_session_bars_for_window(
    ticks_utc: pd.DataFrame, win: WindowDef, tz_pt: str
) -> pd.DataFrame:
    df = ticks_utc.copy()
    pt_naive = _to_pt_naive(df.index, tz_pt)
    df = df.assign(pt_dt=pt_naive.values)
    df["pt_date"] = df["pt_dt"].dt.date

    from datetime import time as dtime
    sh, sm = _parse_hhmm(win.start)
    eh, em = _parse_hhmm(win.end)
    start_t = dtime(sh, sm)
    end_t = dtime(eh, em)

    def in_window(s: pd.Series) -> pd.Series:
        t = s.dt.time
        if start_t <= end_t:
            return (t >= start_t) & (t <= end_t)
        else:
            return (t >= start_t) | (t <= end_t)

    groups = []
    for d, sub in df.groupby("pt_date", sort=True):
        mask = in_window(sub["pt_dt"])
        s = sub.loc[mask]
        if s.empty:
            continue
        o = s["mid"].iloc[0]
        h = s["mid"].max()
        l = s["mid"].min()
        c = s["mid"].iloc[-1]
        # realized variance on 1-minute bars
        spt = s.set_index(pd.DatetimeIndex(s["pt_dt"]).tz_localize(tz_pt))
        m1 = spt["mid"].resample("1T").last().ffill()
        rv = (np.log(m1).diff() ** 2).sum()
        groups.append(
            {"pt_date": pd.to_datetime(str(d)), "open": o, "high": h, "low": l, "close": c, "rv": rv, "tick_count": len(s)}
        )
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
        [c for c in df_cols if c.startswith(f"{wkey}_")] +
        [c for c in df_cols if c.startswith("dow_")] +
        ["is_eom", "is_eoq", f"is_{wkey}"]
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
            [c for c in sub.columns
             if c.startswith(f"{wkey}_") and not (c.endswith("_up") or c.endswith("_pips"))]
            + [c for c in sub.columns if c.startswith("dow_")]
            + ["is_eom", "is_eoq", f"is_{wkey}"]
        )
        # de-dupe preserving order
        seen = set()
        feat_cols = [c for c in feat_cols if not (c in seen or seen.add(c))]
        # save manifest
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

    # ensure plain Python float here
    auc_val = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    grid = np.linspace(0.35, 0.65, 61)
    best = {"acc": (-1, 0.5), "mcc": (-1, 0.5), "pnl": (-1e99, 0.5)}

    for t in grid:
        pred = (p >= t).astype(int)
        acc = (pred == y).mean()
        if acc > best["acc"][0]:
            best["acc"] = (acc, float(t))
        mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else -1.0
        if mcc > best["mcc"][0]:
            best["mcc"] = (float(mcc), float(t))
        dir_pred = np.where(pred == 1, 1.0, -1.0)
        pnl = float(np.sum(dir_pred * true_pips))
        if pnl > best["pnl"][0]:
            best["pnl"] = (pnl, float(t))

    flip_bool = bool(auc_val < 0.5) if np.isfinite(auc_val) else False

    return {
        "auc": auc_val,
        "flip": flip_bool,  # <— now a plain bool
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
        # derive from dataset columns and persist so future runs are consistent
        feat_cols = _cols_for_window(sub.columns, wkey)
        _save_cols(models_dir, wkey, feat_cols)

    X = sub.copy()
    # add any missing columns and enforce order
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

        # Use manifest if present; otherwise derive & save
        X = _get_feat_matrix_for_predict(sub, wkey, models_dir)
        x_last = X.iloc[[-1]]

        cls = load(models_dir / f"model_cls_{wkey}.joblib")
        reg = load(models_dir / f"model_reg_{wkey}.joblib")

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

    # Optional: keep salvage checkpoints if you want
    # feats.to_parquet(Path(cfg["paths"]["feats_only"]))
    # labels.to_parquet(Path(cfg["paths"]["labs_only"]))

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

# ============================== Holdout Eval =================================

def evaluate_holdout(df: pd.DataFrame, cfg: dict, cutoff: pd.Timestamp) -> Dict[str, Dict[str, float]]:
    models_dir = ensure_dir(cfg["paths"]["models_dir"])
    res = {}

    # Split
    train_df = df[df.index < cutoff]
    test_df  = df[df.index >= cutoff]

    # Train and persist on the train slice
    _ = train_and_eval_per_window(train_df, cfg, models_dir)

    # Evaluate on the test slice (guard against NaNs)
    for wkey in ["daily", "w2_10", "w3_12"]:
        sub_tr = train_df[train_df["window"] == wkey]
        sub_te = test_df[test_df["window"] == wkey]
        if sub_tr.empty or sub_te.empty:
            continue

        # Labels
        yte_cls = sub_te[f"{wkey}_up"].astype(float)   # cast to float for nan-safe masks
        yte_reg = sub_te[f"{wkey}_pips"].astype(float)

        # Features from manifest (add missing columns as 0, enforce order)
        feat_cols = _load_cols(models_dir, wkey)
        Xte = sub_te.copy()
        for c in feat_cols:
            if c not in Xte.columns:
                Xte[c] = 0.0
        Xte = Xte[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Load models
        cls = load(models_dir / f"model_cls_{wkey}.joblib")
        reg = load(models_dir / f"model_reg_{wkey}.joblib")

        # ---- Classification metrics (mask only rows with valid yte_cls) ----
        m_cls = yte_cls.notna().to_numpy()
        if m_cls.any():
            proba_up = np.full(len(yte_cls), np.nan, dtype=float)
            proba_up[m_cls] = cls.predict_proba(Xte.iloc[m_cls])[:, 1]
            pred_up = (proba_up[m_cls] >= 0.5).astype(int)
            y_true_cls = yte_cls[m_cls].astype(int).to_numpy()
            auc = float(roc_auc_score(y_true_cls, proba_up[m_cls])) if len(np.unique(y_true_cls)) > 1 else float("nan")
            brier = float(brier_score_loss(y_true_cls, proba_up[m_cls]))
            acc = float(accuracy_score(y_true_cls, pred_up))
        else:
            auc = float("nan"); brier = float("nan"); acc = float("nan")

        # ---- Regression metrics (mask only rows with valid yte_reg) ----
        m_reg = yte_reg.notna().to_numpy()
        if m_reg.any():
            y_pred_reg = reg.predict(Xte.iloc[m_reg])
            mae = float(mean_absolute_error(yte_reg[m_reg], y_pred_reg))
        else:
            mae = float("nan")

        res[wkey] = {
            "AUC": auc,
            "Brier": brier,
            "Acc": acc,
            "MAE_pips": mae,
            "n_train": int(len(sub_tr)),
            "n_test": int(len(sub_te)),
            "cutoff": str(cutoff.date()),
        }

    return res


# ============================== CLI =========================================

def main():
    ap = argparse.ArgumentParser(description="FX pipeline (single-file)")
    ap.add_argument("--config", type=str, default=None, help="Optional YAML config")
    ap.add_argument("--ticks-glob", type=str, default=None, help="Override ticks glob")

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="Ingest ticks→bars→features+labels→dataset parquet")
    sub.add_parser("train", help="Train per-window models, persist manifests + policies")
    ap_pred = sub.add_parser("predict", help="Predict next session/day from latest dataset")
    ap_pred.add_argument("--objective", choices=["acc", "mcc", "pnl"], default="acc",
                         help="Decision rule to use (default: acc)")
    hp = sub.add_parser("holdout", help="Forward holdout: train < cutoff, test ≥ cutoff")
    g = hp.add_mutually_exclusive_group(required=True)
    g.add_argument("--cutoff", type=str, help="Cutoff date YYYY-MM-DD")
    g.add_argument("--last-days", type=int, help="Use last N days as holdout (compute cutoff)")

    args, _ = ap.parse_known_args()
    cfg = load_config(args.config)
    if args.ticks_glob:
        cfg["paths"]["ticks_glob"] = args.ticks_glob
    models_dir = ensure_dir(cfg["paths"]["models_dir"])
    features_p = Path(cfg["paths"]["features_out"])

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
        # Drop labels but keep 'window' to select per-window rows,
        # feature construction is handled by column manifests.
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
        if args.cutoff:
            cutoff = pd.Timestamp(args.cutoff)
        else:
            last_days = int(args.last_days)
            cutoff = df.index.max() - pd.Timedelta(days=last_days)
        res = evaluate_holdout(df, cfg, cutoff)
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
