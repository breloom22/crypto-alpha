"""
Improvement B — break-even stop + extended max-hold.

Once a trade is up by `trigger`, move the stop to entry (lock in "no loss") and
let the position run longer than the base max-hold — the MaxHold-exit diagnosis
showed trades closing at +2.2% on average that still had room.

    B1  +3% -> BE, hold 21d                       (pure break-even floor)
    B2  +3% -> BE + 1xATR trailing, hold 21d       (ride the trend after BE)
    B3  +5% -> BE + 1xATR trailing, hold 14d       (later, tighter*)

* B3 in the spec also scales out half the position; the v3 simulator has no
  partial fills, so it is approximated as break-even + ATR trailing after +5%.
  The approximation is noted in the module-effectiveness report.
"""
from __future__ import annotations

from dataclasses import replace

from strategy import Strategy
from exits import ExitConfig

# (label, be_trigger, extend_max_hold, profit_trail_trigger, profit_trail_atr)
B_VARIANTS = (
    ("B1", 0.03, 21, None, None),
    ("B2", 0.03, 21, 0.03, 1.0),
    ("B3", 0.05, 14, 0.05, 1.0),
)


def _exit_with_be(base: ExitConfig, be: float, hold: int,
                  pt_trig, pt_atr, tag: str) -> ExitConfig:
    """Add a break-even floor (and optional ATR profit-trailing) on top of the
    base exit, extending max-hold. The base's initial SL/TP stay intact until
    BE arms."""
    kw = dict(breakeven_trigger=be, max_hold=hold, label=f"{base.label}+{tag}")
    if pt_trig is not None:
        kw["profit_trail_trigger"] = pt_trig
        kw["profit_trail_atr"] = pt_atr
    return replace(base, **kw)


def variants(base: Strategy) -> "list[Strategy]":
    out = []
    for tag, be, hold, pt_trig, pt_atr in B_VARIANTS:
        out.append(replace(base,
                           id=f"{base.id}+{tag}",
                           exit=_exit_with_be(base.exit, be, hold, pt_trig, pt_atr, tag)))
    return out
