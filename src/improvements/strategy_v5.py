"""
strategy_v5.py — Strategy subclass adding Phase-5 entry-timing knobs.

StrategyV5 keeps v3's entry signals but can:
  * delay entry by N bars (entry_delay)  — dodge the 0~1 day whipsaw losses
  * require a confirmation bar (entry_confirmation)
  * drop chosen assets (exclude_assets)  — e.g. DOGE for S1.11_S, XRP for SEQ
  * inject per-member signal params (member_params) so a SEQ's SHORT leg (NR7 n)
    and LONG leg (NATR / Supertrend) can be tuned independently.

Entry-timing is implemented as a transform of the boolean entry series, so the
unchanged backtester (signal at t -> fill at t+1 open) still applies. All
transforms are look-ahead safe except the gap confirmation, which by definition
inspects the entry bar's own open (the fill price) — noted inline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategy import Strategy
from filters import apply_filters
from signals import registry


def _transform_entry(e: pd.Series, df: pd.DataFrame, direction: str,
                     delay: int, confirmation):
    """Apply confirmation then delay to an entry series (True at bar k => the
    backtester enters at k+1 open)."""
    is_short = direction == "SHORT"
    if confirmation == "close_below_open":
        # require the bar AFTER the signal to move our way, enter the bar after
        cond = (df["close"] < df["open"]) if is_short else (df["close"] > df["open"])
        e = e.shift(1).fillna(False) & cond
    elif confirmation == "gap_down":
        # require a gap our way on the entry bar's open (the fill price itself)
        nxt_open, prev_close = df["open"].shift(-1), df["close"]
        gap = (nxt_open < prev_close) if is_short else (nxt_open > prev_close)
        e = e & gap.fillna(False)
    if delay and delay > 0:
        e = e.shift(delay).fillna(False)
    return e.fillna(False).astype(bool)


@dataclass
class StrategyV5(Strategy):
    entry_delay: int = 0
    entry_confirmation: "str | None" = None
    exclude_assets: tuple = ()
    member_params: dict = field(default_factory=dict)   # sid -> param dict

    def _params_for(self, sid: str):
        return self.member_params.get(sid) or (self.signal_params or None)

    def build_entries(self, data, symbol):
        idx = data[symbol].index
        false = pd.Series(False, index=idx)
        if symbol in self.exclude_assets:
            return false.copy(), false.copy(), false.copy(), false.copy()

        long_e, short_e = false.copy(), false.copy()
        if self.combo_type == "SEQ":
            for sid, d in self.members:
                e = registry.compute_entries(registry.get(sid), data, symbol, d,
                                             self._params_for(sid))
                if d == "LONG":
                    long_e = long_e | e
                else:
                    short_e = short_e | e
        elif self.combo_type == "AND":
            acc = None
            d = self.direction
            for sid, _ in self.members:
                e = registry.compute_entries(registry.get(sid), data, symbol, d,
                                             self._params_for(sid))
                acc = e if acc is None else (acc & e)
            (long_e if d == "LONG" else short_e).loc[:] = acc.fillna(False)
        else:
            sid, d = self.members[0]
            e = registry.compute_entries(registry.get(sid), data, symbol, d,
                                         self._params_for(sid))
            if d == "LONG":
                long_e = e
            else:
                short_e = e

        if self.filters:
            if long_e.any():
                long_e = apply_filters(long_e, self.filters, data, symbol, "LONG")
            if short_e.any():
                short_e = apply_filters(short_e, self.filters, data, symbol, "SHORT")

        df = data[symbol]
        long_e = _transform_entry(long_e.fillna(False), df, "LONG",
                                  self.entry_delay, self.entry_confirmation)
        short_e = _transform_entry(short_e.fillna(False), df, "SHORT",
                                   self.entry_delay, self.entry_confirmation)
        return long_e, short_e, false.copy(), false.copy()
