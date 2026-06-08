"""
improvements/ — Phase 4 strategy-optimization modules.

Each module improves the trade-level behaviour of a *base* v3 strategy without
touching its entry-signal logic (spec "고치는 것이지 새로 만드는 것이 아니다"):

  dynamic_sl_tp   (A) — ATR-based stop/target instead of fixed percent
  break_even      (B) — break-even stop + extended max-hold
  whipsaw_guard   (C) — extra entry filters that dodge chop / loss streaks
  asset_weighting (D) — drop / weight assets by historical strength
  alt_long_signals(E) — alternative LONG legs for the SEQ strategies
  stepped_trailing(F) — two-tier ATR trailing stop
  portfolio       (G) — combine several strategies into one book

All builders take a base :class:`Strategy` (or :class:`ExitConfig`) and return a
new object; the base is never mutated. The same v3 backtester / scorer / OOS /
walk-forward / random framework then re-validates the result.
"""
