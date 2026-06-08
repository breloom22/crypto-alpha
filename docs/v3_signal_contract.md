# v3 Signal Module Contract (FROZEN)

This is the binding interface for every signal module in `src/signals/`. Follow it
exactly. The foundation files it references already exist and are tested — do NOT
modify them; only call them.

## Run/import context
- Working directory for tests: `C:\AI_AGENTS\NewStrategy\src`. Run tests with
  `cd C:\AI_AGENTS\NewStrategy\src && python -c "..."`.
- Imports inside a signal module:
  ```python
  from indicators import _ta
  from indicators import _ta_extended as tae
  from indicators import _patterns as pat            # S1 only
  from indicators.base import market_frame, market_index, ASSETS, SECTORS  # S7 only
  from .registry import signal, LONG, SHORT, BOTH
  import numpy as np
  import pandas as pd
  ```

## Data format
`data: dict[str, pd.DataFrame]`. Each df is one asset, DatetimeIndex (daily),
columns: `open, high, low, close, volume` (float) + `outlier` (bool). Assets:
BTC, ETH, SOL, DOGE, OP, AVAX, XRP, XLM, SUI. NOTE: this dataset has
`open[t] == close[t-1]` (no overnight gaps), so gap signals will simply rarely/never
fire — implement them faithfully anyway.

## Signal interface
Per-asset signal (groups S1–S6, S8):
```python
@signal("S2.03", "Connors RSI", "S2", BOTH, rsi_period=3)   # defaults are kwargs
def connors_rsi(df, direction, rsi_period=3, streak_period=2, rank_period=100):
    crsi = tae.connors_rsi(df["close"], rsi_period, streak_period, rank_period)
    return crsi < 15 if direction == "LONG" else crsi > 85   # boolean pd.Series
```
- `df` is ONE asset's OHLCV frame; return a boolean `pd.Series` aligned to `df.index`.
- A True at day t means "enter at t+1 open". The backtester handles entry timing,
  non-overlap, and cooldown — you only produce the boolean trigger series.
- `direction` is `"LONG"` or `"SHORT"`. Declare available directions in the
  decorator's 4th arg: `BOTH`, `(LONG,)`, or `(SHORT,)`.

Cross-asset signal (group S7 only):
```python
@signal("S7.01", "ALT/BTC RS reversal", "S7", (LONG,), cross_asset=True)
def alt_btc_rs(data, symbol, direction, period=14):
    if symbol == "BTC":
        return pd.Series(False, index=data["BTC"].index)   # ratio vs self is trivial
    ratio = data[symbol]["close"] / data["BTC"]["close"].reindex(data[symbol].index)
    r = tae.rsi(ratio, period)
    return _ta.crosses_above(r, 30)
```
- Signature is `(data, symbol, direction, **params)`; return the entry series for
  `symbol`. You may read every asset to compute it. Pass `cross_asset=True`.

## LOOK-AHEAD SAFETY (mandatory)
The value at index t may depend ONLY on data at indices <= t. Use rolling/ewm/
shift(+k) (all backward-looking). NEVER use `.shift(-k)`, centered windows,
full-series `.max()/.min()/.quantile()` used as a per-bar threshold, or future
data. To compare today vs a prior window, shift the rolling stat:
`prior_high = high.rolling(n).max().shift(1)`. Entry at t+1 open is what makes
the t-th signal tradeable — do not "peek" at t+1.

## Threshold crosses vs zones
- Wording like "크로스업/크로스다운" → use `_ta.crosses_above(series, level)` /
  `_ta.crosses_below(series, level)` (edge-triggered; level may be a scalar or Series).
- Zone conditions ("CRSI < 15", "MFI > 80") → return the raw boolean; the
  backtester de-duplicates via non-overlap + cooldown, so continuous firing is fine.
- "음→양 전환" (sign flip to positive) → `(x > 0) & (x.shift(1) <= 0)`.

## Available foundation API

### `from indicators import _ta` (base primitives)
`sma(s,n)`, `ema(s,n)`, `rsi(close,n=14)`, `roc(close,n=10)`,
`macd(close,fast=12,slow=26,signal=9)->(line,signal,hist)`,
`stochastic(high,low,close,k=14,d=3)->(k,d)`, `cci(high,low,close,n=20)`,
`williams_r(high,low,close,n=14)`, `true_range(h,l,c)`, `atr(h,l,c,n=14)`,
`bollinger(close,n=20,k=2.0)->(mid,upper,lower,width)`,
`keltner(h,l,c,n=20,atr_mult=2.0)->(mid,upper,lower)`,
`adx(h,l,c,n=14)->(plus_di,minus_di,adx)`,
`ichimoku(h,l,tenkan=9,kijun=26,senkou=52)->(conv,base,span_a,span_b)`,
`obv(close,volume)`, `cmf(h,l,c,v,n=20)`, `zscore(s,n)`,
`crosses_above(a,b)`, `crosses_below(a,b)`, `rolling_argmax_is_now(s,n)`.

### `from indicators import _ta_extended as tae`
Re-exports `sma, ema, rsi, roc, atr, true_range, zscore, bollinger, keltner`.
Helpers: `wma(s,n)`, `rolling_percentile(s,n)->0..1`, `rolling_mad(s,n)`,
`consecutive_directional(close)-> signed run length (+k up / -k down)`.

S2 momentum:
`fisher_transform(high,low,period=10)`,
`inverse_fisher_rsi(close,rsi_period=14) -> (-1,1)`,
`connors_rsi(close,rsi_period=3,streak_period=2,rank_period=100) -> 0..100`,
`tsi(close,r=25,s=13,signal=7) -> (tsi, signal)`,
`coppock(close,roc1=14,roc2=11,wma_period=10)`,
`kst(close) -> (kst, signal)`,
`aroon(high,low,period=25) -> (up, down, oscillator)`,
`vortex(high,low,close,n=14) -> (plus_vi, minus_vi)`,
`elder_ray(high,low,close,period=13) -> (bull_power, bear_power, ema)`,
`cmo(close,period=14) -> -100..100`,
`dpo(close,period=20)`,
`ultimate_oscillator(high,low,close,p1=7,p2=14,p3=28) -> 0..100`,
`stoch_rsi(close,rsi_period=14,stoch_period=14) -> 0..1`,
`rvi(open_,high,low,close,period=4) -> (rvi, signal)`,
`mass_index(high,low,ema_period=9,sum_period=25)`,
`random_walk_index(high,low,close,n=14) -> (rwi_high, rwi_low)`.

S3 trend:
`dema(close,n)`, `tema(close,n)`, `hull_ma(close,n=20)`,
`kama(close,er_period=10,fast=2,slow=30)`, `mcginley(close,n=14)`,
`vidya(close,cmo_period=9,n=12)`, `linreg_slope(close,n=20)`,
`linreg_value(close,n=20)`, `linreg_channel(close,n=20,k=2.0) -> (mid,upper,lower)`,
`heikin_ashi(open_,high,low,close) -> (ha_open,ha_high,ha_low,ha_close)`,
`supertrend(high,low,close,period=10,mult=3.0) -> (supertrend, direction)`  # direction +1 up / -1 down
`pivot_points(high,low,close) -> (pp, r1, s1)`  # from PREVIOUS bar.

S4 volatility:
`ttm_squeeze(high,low,close,bb_n=20,bb_k=2,kc_n=20,kc_mult=1.5) -> (squeeze_on_bool, momentum)`,
`chaikin_volatility(high,low,n=10)`, `hist_vol(close,n=20)`,
`natr(high,low,close,n=14)`, `yang_zhang_vol(open_,high,low,close,n=20)`,
`parkinson_vol(high,low,n=20)`, `ulcer_index(close,n=14)`,
`atr_trailing_stop(high,low,close,n=14,mult=3) -> (stop, direction)`,
`chandelier_exit(high,low,close,n=22,mult=3) -> (long_stop, short_stop)`.
For BB %B use `_ta.bollinger`; for Keltner-BB spread use `_ta.bollinger` + `_ta.keltner`.

S5 volume:
`mfi(high,low,close,volume,n=14) -> 0..100`,
`ease_of_movement(high,low,volume,n=14) -> (emv, sma_emv)`,
`klinger(high,low,close,volume,fast=34,slow=55,signal=13) -> (kvo, signal)`,
`nvi(close,volume,signal=255) -> (nvi, signal)`,
`pvi(close,volume,signal=255) -> (pvi, signal)`,
`volume_oscillator(volume,fast=5,slow=20)`,
`vwap_rolling(high,low,close,volume,n=20)`,
`ad_line(high,low,close,volume)`,
`ad_oscillator(high,low,close,volume,fast=3,slow=10)`,
`rvol(volume,n=20)`.

S6 statistical:
`hurst_exponent(close,n=60,max_lag=20)`, `rolling_autocorr(returns,n=20,lag=1)`,
`variance_ratio(close,k=5,n=60)`, `return_entropy(returns,n=30,bins=10)`,
`clv(high,low,close) -> -1..1`, `distance_from_ma(close,n=50)`,
`median_reversion(close,n=50)`, plus `zscore`, `rolling_percentile`,
`consecutive_directional`. For skew/kurt use pandas:
`close.pct_change().rolling(n).skew()` / `.kurt()`.

### S1 patterns — `from indicators import _patterns as pat`
`bullish_engulfing(df)`, `bearish_engulfing(df)`, `hammer_shape(df)`,
`shooting_star_shape(df)`, `doji(df,thresh=0.1)`, `inside_bar(df)`,
`outside_bar(df)`, `three_white_soldiers(df)`, `three_black_crows(df)`,
`morning_star(df)`, `evening_star(df)`, `pin_bar_bullish(df)`,
`pin_bar_bearish(df)`, `gap_up(df)`, `gap_down(df)`,
`donchian_high_break(df,n=20)`, `donchian_low_break(df,n=20)`, `nr7(df,n=7)`,
`range_contraction(df,short_n=3,long_n=20,frac=0.5)`,
`range_expansion(df,lookback=1,mult=2.0)`.
Anatomy helpers: `body(df)`, `candle_range(df)`, `upper_wick(df)`,
`lower_wick(df)`, `is_bull(df)`, `is_bear(df)`, `close_location(df) -> 0..1`,
`roc_pct(close,n=5)`, `uptrend(close,n=5,thresh=3.0)`, `downtrend(close,n=5,thresh=3.0)`.

## Module requirements
- One module per group, file names: `price_action.py` (S1), `alt_momentum.py` (S2),
  `alt_trend.py` (S3), `volatility.py` (S4), `alt_volume.py` (S5),
  `statistical.py` (S6), `cross_asset.py` (S7), `nonstandard.py` (S8).
- Implement EVERY row in your group's table with the EXACT id (e.g. `S2.03`) and a
  clear name. Use `BOTH` when the table gives both LONG and SHORT conditions; use
  `(LONG,)` / `(SHORT,)` when only one side is defined (e.g. several S7 rows).
- Each `@signal`-decorated function must return a boolean Series (per-asset) for
  the requested direction. Guard against div-by-zero (`.replace(0, np.nan)`).
- Registration happens at import time via the decorator — no `register_all()` needed.
- The module must import cleanly and every signal must produce a boolean Series on
  real BTC/ETH data with no exceptions and no all-True/all-False degeneracy for the
  zone signals (a handful of fires is expected; some rare patterns may be sparse).

## Self-test each function before finishing
```python
cd C:\AI_AGENTS\NewStrategy\src
python -c "
from utils import setup_console; setup_console()
from data_loader import load_data
from signals import <your_module> as m
from signals import registry
data = load_data('../data')
for s in [x for x in registry.all_specs() if x.group=='<SX>']:
    for d in s.directions:
        if s.cross_asset:
            for sym in data:
                sig = registry.compute_entries(s, data, sym, d)
        else:
            sig = registry.compute_entries(s, data, 'BTC', d)
        assert sig.dtype==bool, (s.id,d)
    print(s.id, s.name, 'OK')
"
```
