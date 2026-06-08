# Crypto Crash Detector

OHLCV-only search for **leading technical signals before major crypto crashes**, with an
objective crash-event definition, a 35-indicator candidate set, a leakage-safe backtest, ensemble
search, in/out-of-sample validation, and an interactive dashboard.

> Motivating event: the ~2026-06-04 multi-asset plunge (BTC −17% over a week). The pipeline
> isolates it as a distinct crash leg across all 9 assets and tests whether it (and 260+ other
> historical legs) was preceded by repeatable warning signals.

## Quick start

```bash
pip install -r requirements.txt
python main.py            # downloads data if missing, runs the whole pipeline
# open results/dashboard.html
```

Options: `python main.py --download` (force re-download), `python main.py --quick`
(skip in/out-sample + sensitivity). Individual phases are runnable too, e.g.
`python src/crash_detector.py`, `python src/scorer.py`, `python src/ensemble.py`.

## Pipeline

| Phase | Module | Output |
|------|--------|--------|
| 0 Data | `src/data_loader.py` | `data/{SYM}_daily_ohlcv.csv`, `all_ohlcv.csv` |
| 1 Crash events | `src/crash_detector.py` | `results/crash_events.csv`, `crash_clusters.csv`, `simultaneous_crashes.csv` |
| 2 Indicators (35) | `src/indicators/*` | — |
| 3 Backtest | `src/backtester.py` | `results/backtest_full.csv` |
| 4 Scoring & ranking | `src/scorer.py` | `results/individual_scores.csv` |
| 6/7 Validation | `src/scorer.py` | `results/in_out_sample.csv`, `sensitivity.csv` |
| 5 Ensembles | `src/ensemble.py` | `results/ensemble_scores.csv` |
| 6 Dashboard | `src/visualizer.py` | `results/dashboard.html` (7 charts) |

`main.py` runs all phases with an integrity gate between each.

## Crash-event definition (Phase 1)

Four parallel criteria, any one fires an event **on the day the condition first crosses true**
(onset-edge, not the persistent state):

- **CRASH_A** 7-day return ≤ −15% · **CRASH_B** 3-day return ≤ −10%
- **CRASH_C** drawdown ≥ 20% from the 52-week high · **CRASH_D** 1-day return ≤ −8%

Event days within 14 days are merged into one **crash leg**; the leg's first day is the onset a
leading signal must precede.

## Indicators (Phase 2) — 35 candidates

- **Momentum (8)** RSI overbought-exit, RSI bearish divergence, MACD dead-cross, MACD-hist flip,
  Stochastic cross, CCI reversal, Williams %R, ROC turn.
- **Trend (7)** EMA 20/50, SMA 50/200, EMA200 break, ADX DI cross, Parabolic-SAR flip,
  Lower-High+Lower-Low, Ichimoku cloud breakdown.
- **Volatility (7)** Bollinger lower break, BB squeeze→expansion, ATR spike, Garman-Klass spike,
  range expansion, Keltner lower break, consecutive down candles.
- **Volume (6)** volume-z spike on down day, OBV divergence, CMF flip, VPT breakdown,
  Force-Index flip, down/up volume ratio.
- **Cross-asset / market-level (7)** alt-vs-BTC weakness, multi-asset RSI overbought, correlation
  spike, breadth deterioration, all-sector weakness, volume-weighted momentum, volatility regime shift.

## Backtest metrics (Phase 3)

A signal is a leading warning if a crash onset follows it within **[1, 21] days**.
Per indicator × asset × crash-type: **precision** (signal-centric), **recall** (crash-centric),
**F1**, **avg/median lead time**, **false-alarm rate**, and per-CRASH-type recall.

`composite = 0.30·precision + 0.25·recall + 0.20·(1−false_alarm) + 0.15·norm_lead + 0.10·consistency`
where consistency = share of the 9 assets with F1 > 0.3.

## Key findings

Numbers from the latest run (2022-01-01 → 2026-06-06, 9 assets, 267 crash legs).

- **Base rate ≈ 0.39.** Crypto crashes are frequent: a *randomly-timed* signal is followed by a
  crash onset within 21 days ~39% of the time. So **precision-lift over the base rate**, not raw
  precision/recall/F1, is the honest discriminator.
- **Only momentum-exhaustion signals beat random on precision.** `M5` Stochastic %K/%D bearish
  cross above 80 is the single best (precision 0.49 ≈ **1.27× base rate**, F1 0.58, ~10-day lead);
  `M7` Williams %R and `M6` CCI reversal also clear ~1.2×.
- **High-recall ≠ informative.** `VOL6`, `V7`, `X5` rank well on composite/F1 because they fire
  often and catch ~80% of crashes, but their precision (~0.33–0.35) is **at or below the base
  rate** — recall-driven, little predictive edge. The dashboard's base-rate line makes this visible.
- **Best ensemble** `M5+M8+M6` (OR): F1 0.60, recall 0.96, but precision 0.44 (~1.1× base) — OR
  maximizes recall at the cost of precision; only a modest lift over the best single indicator.
- **Holds out-of-sample:** 71% of indicators keep their rank within ±5 between 2022–24 and
  2025–26, and the top momentum signals are robust to ±20% parameter changes.
- **Takeaway:** there is a *weak but real* leading signal in overbought-rollover / momentum-
  exhaustion indicators (~10-day lead, ~1.2–1.3× precision over base rate). No single OHLCV
  indicator is a reliable crash alarm — the honest result is a modest edge, not a siren.

## Design decisions

- **No `pandas_ta`.** It is incompatible with numpy ≥ 2.0 (imports the removed `numpy.NaN`).
  All indicators are implemented directly in vectorized pandas/numpy in `src/indicators/_ta.py`,
  which also gives full control over look-ahead safety.
- **Onset-edge crash events.** The literal "drawdown ≥ 20%" level condition stays true for months
  in bear markets and, with 14-day clustering, collapses entire multi-year declines into one giant
  leg — hiding distinct crashes (including the 2026-06 plunge). Edge-triggering each criterion
  (`state(t) ∧ ¬state(t−1)`) yields discrete, correctly-dated legs and is look-ahead safe.
- **Unified signal interface.** Per-asset and market-level (X) indicators both expose
  `signals(...) → bool Series`; the X-series is evaluated against every asset's crashes, so one
  framework scores all 35.
- **Data through 2026-06-06.** Extends slightly past the spec's 2026-05-31 to capture the
  motivating crash; the in/out-sample split (in 2022–24, out 2025–26) follows the spec.

## Look-ahead safety

Every indicator passes a **truncation-invariance test** (`src/_test_indicators.py`): truncating the
data must not change any earlier signal value. All TA primitives use only backward-looking
`rolling`/`ewm`/`shift(+k)`. The Ichimoku senkou spans use the standard +26 displacement (the cloud
at *t* derives from data at *t−26*, i.e. known information); the chikou span is deliberately omitted.

## Caveats

- This is a **statistical signal study, not a trading system** — no slippage, fees, or liquidity.
- **Overfitting / multiple testing:** 35 indicators × parameter sweeps can surface chance patterns;
  always read results alongside the in/out-sample stability and sensitivity tables.
- **Base-rate context:** crypto crashes are frequent and the 21-day window is wide, so high recall
  is relatively easy — precision (and lift over the base rate) is the discriminating metric.
- **Survivorship / listing dates:** OP (2022-06) and SUI (2023-05) have shorter histories; cross-asset
  breadth counts only listed assets per day.
