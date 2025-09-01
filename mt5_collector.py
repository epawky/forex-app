# mt5_collector.py
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd

# Import MetaTrader5 lazily so this file can be linted without MT5 installed
try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed. Run: pip install MetaTrader5", file=sys.stderr)
    sys.exit(1)

def _to_utc(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def init_mt5():
    """
    Initialize the MT5 terminal connection.
    Supports two modes:
      1) Reuse an already logged-in terminal.
      2) Initialize with credentials + server + terminal path from env vars.
    """
    term_path = os.getenv("MT5_TERMINAL_PATH")  # e.g. C:\Program Files\MetaTrader 5\terminal64.exe
    login     = os.getenv("MT5_LOGIN")          # e.g. 12345678
    password  = os.getenv("MT5_PASSWORD")       # e.g. ********
    server    = os.getenv("MT5_SERVER")         # e.g. 'YourBroker-Server'

    kwargs = {}
    if term_path: kwargs["path"] = term_path
    if login and password and server:
        kwargs.update(dict(login=int(login), password=password, server=server))

    if not mt5.initialize(**kwargs):
        err = mt5.last_error()
        raise RuntimeError(f"MT5 initialize failed: {err}")
    return True

def shutdown_mt5():
    try:
        mt5.shutdown()
    except Exception:
        pass

def load_last_timestamp(csv_path: Path) -> datetime | None:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty or "time" not in df.columns:
            return None
        tmax = pd.to_datetime(df["time"], utc=True, errors="coerce").max()
        if pd.isna(tmax):
            return None
        return tmax.to_pydatetime()
    except Exception:
        return None

def fetch_ticks(symbol: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: time(ISO UTC), bid, ask, mid
    """
    start = int(_to_utc(start_utc).timestamp())
    end   = int(_to_utc(end_utc).timestamp())

    # MetaTrader returns numpy structured array; 'time' is UTC epoch seconds
    ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return pd.DataFrame(columns=["time", "bid", "ask", "mid"])

    df = pd.DataFrame(ticks)
    # Some brokers return NaN for ask or bid for trade-only ticks; drop those
    if "bid" not in df.columns: df["bid"] = pd.NA
    if "ask" not in df.columns: df["ask"] = pd.NA
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df = df.dropna(subset=["bid", "ask"])
    if df.empty:
        return pd.DataFrame(columns=["time", "bid", "ask", "mid"])

    # Convert epoch seconds to UTC ISO string
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    # Export as strings for CSV consistency
    out = df[["time", "bid", "ask", "mid"]].copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f%z").str.replace("+0000", "Z", regex=False)
    return out

def append_csv(csv_path: Path, df: pd.DataFrame):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = not csv_path.exists()
    df.to_csv(csv_path, index=False, mode="a", header=header)

def main():
    ap = argparse.ArgumentParser(description="Fetch ticks from MetaTrader 5 and append CSV.")
    ap.add_argument("--symbol", default=os.getenv("MT5_SYMBOL", "EURUSD"))
    ap.add_argument("--out-dir", default=str(Path("data") / "ticks"))
    ap.add_argument("--days", type=int, default=1, help="Days to backfill if no existing file")
    ap.add_argument("--file", default=None, help="Optional explicit output filename")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.file:
        out_file = out_dir / args.file
    else:
        # daily file naming is simple and keeps repos clean
        out_file = out_dir / f"{args.symbol}_ticks_{datetime.utcnow():%Y-%m-%d}.csv"

    try:
        init_mt5()
        # Figure start/end
        last_ts = load_last_timestamp(out_file)
        end_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
        if last_ts is None:
            start_utc = end_utc - timedelta(days=args.days)
        else:
            start_utc = last_ts + timedelta(milliseconds=1)

        if start_utc >= end_utc:
            print("[mt5_collector] nothing new to fetch.")
            return

        df = fetch_ticks(args.symbol, start_utc, end_utc)
        if df.empty:
            print("[mt5_collector] fetched 0 rows (broker returned no ticks).")
            return

        append_csv(out_file, df)
        print(f"[mt5_collector] wrote {len(df)} ticks -> {out_file}")

    finally:
        shutdown_mt5()

if __name__ == "__main__":
    main()
