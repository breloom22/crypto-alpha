"""Smoke + invariant test for all 35 indicators."""
from utils import setup_console, DATA_DIR
setup_console()
import numpy as np
import pandas as pd
from data_loader import load_data
from indicators.base import build_indicators

data = load_data(DATA_DIR)
reg = build_indicators()
print(f"Registered {len(reg)} indicators: {list(reg)}")
assert len(reg) == 35, f"expected 35, got {len(reg)}"

btc = data["BTC"]
rows = []
problems = []

for iid, ind in reg.items():
    try:
        if ind.cross_asset:
            sig = ind.signals(data)
            # evaluate on BTC dates for the count
            sig_btc = sig.reindex(btc.index).fillna(False)
        else:
            sig = ind.signals(btc)
            sig_btc = sig
        n = int(sig_btc.sum())
        assert sig.dtype == bool, f"{iid} not bool"
        assert not sig.isna().any(), f"{iid} has NaN"
        rows.append((iid, ind.category, ind.name, n, round(100 * n / len(sig_btc), 1)))
        if n == 0:
            problems.append(f"{iid} fired 0 signals on BTC")
        if n > 0.5 * len(sig_btc):
            problems.append(f"{iid} fired on {n}/{len(sig_btc)} days (>50%! likely persistent)")
    except Exception as e:
        problems.append(f"{iid} ERROR: {type(e).__name__}: {e}")
        rows.append((iid, ind.category, getattr(ind, "name", "?"), -1, -1))

df = pd.DataFrame(rows, columns=["id", "category", "name", "btc_signals", "pct_days"])
print("\n=== BTC signal counts per indicator ===")
print(df.to_string(index=False))

# ---- look-ahead leakage test -------------------------------------------------
# Truncating the series must not change earlier signal values.
print("\n=== Look-ahead leakage test (truncation invariance) ===")
leak = []
cutoff = len(btc) - 60
btc_trunc = btc.iloc[:cutoff]
for iid, ind in reg.items():
    if ind.cross_asset:
        full = ind.signals(data).reindex(btc.index).fillna(False).iloc[:cutoff]
        trunc_data = {s: d.iloc[:cutoff] if s == "BTC" else d.loc[:btc.index[cutoff - 1]] for s, d in data.items()}
        tr = ind.signals(trunc_data).reindex(btc_trunc.index).fillna(False)
        full = full.reindex(tr.index).fillna(False)
    else:
        full = ind.signals(btc).iloc[:cutoff]
        tr = ind.signals(btc_trunc)
    # compare overlapping region, ignore last `warmup` edge effects near cutoff
    cmp_n = cutoff - 5
    if not full.iloc[:cmp_n].equals(tr.iloc[:cmp_n]):
        diff = int((full.iloc[:cmp_n].values != tr.iloc[:cmp_n].values).sum())
        leak.append(f"{iid}: {diff} differing signals after truncation -> LOOK-AHEAD?")

if leak:
    print("LEAKAGE SUSPECTED:")
    for x in leak:
        print("  " + x)
else:
    print("OK — all indicators truncation-invariant (no look-ahead detected).")

print("\n=== Problems ===")
if problems:
    for p in problems:
        print("  - " + p)
else:
    print("  none")
