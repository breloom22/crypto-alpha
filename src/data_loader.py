"""
data_loader.py — Phase 0: data acquisition & preprocessing.

Downloads daily OHLCV for the 9 target assets from Binance (public API, no key
required) via ccxt, with OKX/Bybit as automatic fallbacks. Cleans the data
(gap handling, volume-0 removal, outlier flagging) and exposes a uniform
``load_data`` interface for the rest of the pipeline.

Usage:
    python src/data_loader.py --download           # download + clean -> data/
    python src/data_loader.py --download --end 2026-06-06

Programmatic:
    from data_loader import load_data
    data = load_data("data")          # {symbol: DataFrame(index=date, OHLCV)}
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "OP", "AVAX", "XRP", "XLM", "SUI"]

# Known Binance USDT-pair listing dates (used only for logging / sanity checks).
LISTING_DATES = {
    "BTC": "2017-08-17", "ETH": "2017-08-17", "SOL": "2020-08-11",
    "DOGE": "2019-07-05", "OP": "2022-06-01", "AVAX": "2020-09-22",
    "XRP": "2018-05-04", "XLM": "2019-01-31", "SUI": "2023-05-03",
}

DEFAULT_START = "2022-01-01"
DEFAULT_END = "2026-06-06"          # captures the 2026-06-04 crash of interest
EXCHANGE_PRIORITY = ["binance", "okx", "bybit"]

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# Preprocessing thresholds (from spec)
MAX_GAP_DAYS = 5          # >= this many consecutive missing days -> drop segment
OUTLIER_RET = 0.50        # |daily return| > 50% -> flag as outlier


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------
def _ms(date_str: str) -> int:
    """ISO date 'YYYY-MM-DD' -> epoch milliseconds (UTC midnight)."""
    return int(datetime.strptime(date_str, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def _make_exchange(exchange_id: str):
    import ccxt
    return getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 30000})


def _fetch_paginated(ex, pair: str, since_ms: int, end_ms: int,
                     timeframe: str = "1d") -> list:
    """Page through fetch_ohlcv until ``end_ms``. Returns raw ohlcv rows."""
    day_ms = 24 * 60 * 60 * 1000
    all_rows: list = []
    cursor = since_ms
    while cursor < end_ms:
        batch = ex.fetch_ohlcv(pair, timeframe, since=cursor, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:            # no forward progress -> stop
            break
        cursor = last_ts + day_ms
        if len(batch) < 1000:            # exhausted available history
            break
        time.sleep(max(ex.rateLimit, 50) / 1000.0)
    # de-dup by timestamp, keep within [since, end]
    seen = {}
    for r in all_rows:
        if since_ms <= r[0] <= end_ms:
            seen[r[0]] = r
    return [seen[k] for k in sorted(seen)]


def download_symbol(symbol: str, start: str, end: str,
                    exchanges=EXCHANGE_PRIORITY) -> pd.DataFrame | None:
    """Download one symbol's daily OHLCV, trying exchanges in priority order."""
    since_ms, end_ms = _ms(start), _ms(end)
    last_err = None
    for exid in exchanges:
        try:
            ex = _make_exchange(exid)
            ex.load_markets()
            pair = f"{symbol}/USDT"
            if pair not in ex.symbols:
                last_err = f"{pair} not on {exid}"
                continue
            rows = _fetch_paginated(ex, pair, since_ms, end_ms)
            if not rows:
                last_err = f"{exid} returned no rows"
                continue
            df = pd.DataFrame(rows, columns=["ts", *OHLCV_COLS])
            df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
            df = df.drop(columns="ts").set_index("date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            print(f"  [{symbol:5}] {len(df):4d} rows from {exid:8} "
                  f"({df.index.min().date()} -> {df.index.max().date()})")
            return df
        except Exception as e:                       # noqa: BLE001
            last_err = f"{exid}: {type(e).__name__}: {str(e)[:80]}"
            continue
    print(f"  [{symbol:5}] FAILED — {last_err}")
    return None


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def preprocess(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Clean a single asset's OHLCV frame.

    - reindex to a continuous daily calendar
    - forward-fill gaps shorter than MAX_GAP_DAYS; drop rows inside longer gaps
    - drop rows with volume == 0
    - flag |daily return| > OUTLIER_RET as ``outlier`` (kept, not dropped)
    """
    df = df.copy().sort_index()
    df = df[~df.index.duplicated(keep="last")]

    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full)
    df.index.name = "date"

    # locate consecutive-missing runs (close is NaN)
    missing = df["close"].isna()
    if missing.any():
        run_id = (missing != missing.shift()).cumsum()
        drop_idx = []
        for _, grp in df[missing].groupby(run_id[missing]):
            if len(grp) >= MAX_GAP_DAYS:
                drop_idx.extend(grp.index)        # long gap -> exclude segment
        df = df.ffill()                            # ffill short gaps
        if drop_idx:
            df = df.drop(index=drop_idx)
            print(f"  [{symbol:5}] dropped {len(drop_idx)} rows in >= {MAX_GAP_DAYS}d gaps")

    # drop any rows still NaN (e.g. leading) and volume==0 days
    df = df.dropna(subset=OHLCV_COLS)
    n0 = int((df["volume"] == 0).sum())
    if n0:
        print(f"  [{symbol:5}] dropped {n0} zero-volume rows")
        df = df[df["volume"] > 0]

    # outlier flag (look-ahead safe: uses only past close). Ignore returns that
    # span a dropped/long-gap segment so a discontinuity isn't mis-flagged.
    ret = df["close"].pct_change()
    contiguous = df.index.to_series().diff().dt.days == 1
    ret = ret.where(contiguous)
    df["outlier"] = (ret.abs() > OUTLIER_RET).fillna(False)
    n_out = int(df["outlier"].sum())
    if n_out:
        print(f"  [{symbol:5}] flagged {n_out} outlier days (|ret|>{OUTLIER_RET:.0%})")

    return df[[*OHLCV_COLS, "outlier"]].astype(
        {c: "float64" for c in OHLCV_COLS} | {"outlier": "bool"})


# ----------------------------------------------------------------------------
# Orchestration: download_all + load_data
# ----------------------------------------------------------------------------
def download_all(symbols=SYMBOLS, start=DEFAULT_START, end=DEFAULT_END,
                 data_dir="data") -> dict[str, pd.DataFrame]:
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading {len(symbols)} assets {start} -> {end} ...")
    out: dict[str, pd.DataFrame] = {}
    combined = []
    for sym in symbols:
        raw = download_symbol(sym, start, end)
        if raw is None or raw.empty:
            continue
        clean = preprocess(raw, sym)
        path = os.path.join(data_dir, f"{sym}_daily_ohlcv.csv")
        clean.to_csv(path)
        out[sym] = clean
        c = clean.reset_index()
        c.insert(1, "symbol", sym)
        combined.append(c)
    if combined:
        allc = pd.concat(combined, ignore_index=True)
        allc.to_csv(os.path.join(data_dir, "all_ohlcv.csv"), index=False)
        print(f"Saved per-symbol CSVs + all_ohlcv.csv ({len(allc)} rows) to {data_dir}/")
    return out


def load_data(data_dir="data", symbols=SYMBOLS,
              download_if_missing=True, start=DEFAULT_START,
              end=DEFAULT_END) -> dict[str, pd.DataFrame]:
    """Load cleaned per-symbol OHLCV frames. Downloads if files are absent."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = os.path.join(data_dir, f"{sym}_daily_ohlcv.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
            if "outlier" not in df.columns:
                df["outlier"] = False
            out[sym] = df
    missing = [s for s in symbols if s not in out]
    if missing and download_if_missing:
        print(f"Missing {missing} — downloading ...")
        dl = download_all(missing, start, end, data_dir)
        out.update(dl)
    if not out:
        raise RuntimeError("No data available. Run with --download first.")
    return {s: out[s] for s in symbols if s in out}


def integrity_report(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One-row-per-asset summary used for the data-integrity gate."""
    rows = []
    for sym, df in data.items():
        ret = df["close"].pct_change()
        rows.append({
            "symbol": sym, "rows": len(df),
            "start": df.index.min().date(), "end": df.index.max().date(),
            "outliers": int(df["outlier"].sum()),
            "min_ret": round(float(ret.min()), 3),
            "max_ret": round(float(ret.max()), 3),
            "any_nan": bool(df[OHLCV_COLS].isna().any().any()),
        })
    return pd.DataFrame(rows)


def main():
    try:
        from utils import setup_console
        setup_console()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Download & preprocess crypto OHLCV")
    ap.add_argument("--download", action="store_true", help="force download")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    if args.download:
        data = download_all(SYMBOLS, args.start, args.end, args.data_dir)
    else:
        data = load_data(args.data_dir, download_if_missing=True,
                         start=args.start, end=args.end)

    print("\n=== Data integrity report ===")
    rep = integrity_report(data)
    print(rep.to_string(index=False))
    assert not rep["any_nan"].any(), "NaNs remain in OHLCV!"
    print(f"\nOK: {len(data)} assets loaded.")


if __name__ == "__main__":
    main()
