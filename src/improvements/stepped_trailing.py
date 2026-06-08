"""
Improvement F — two-tier stepped ATR trailing stop.

Replace fixed SL/TP/trailing with an ATR trailing stop that *tightens* once the
trade is comfortably in profit: trail loosely (base_mult x ATR) to let the move
breathe, then snug up (tight_mult x ATR) after +tighten_at to protect gains.
Max-hold is kept from the base. Implemented by ExitConfig.stepped_trail_* and
the E13 branch in backtester_v3._simulate_one.

    F1  base 2.0  tight 1.0  at +5%
    F2  base 2.5  tight 1.5  at +7%
    F3  base 1.5  tight 0.7  at +3%   (aggressive)
"""
from __future__ import annotations

from dataclasses import replace

from strategy import Strategy
from exits import ExitConfig

# (label, base_mult, tight_mult, tighten_at)
F_VARIANTS = (
    ("F1", 2.0, 1.0, 0.05),
    ("F2", 2.5, 1.5, 0.07),
    ("F3", 1.5, 0.7, 0.03),
)


def _exit_stepped(base: ExitConfig, b: float, t: float, at: float, tag: str) -> ExitConfig:
    return replace(
        base,
        stop_loss_pct=None, take_profit_pct=None, trailing_pct=None,
        stepped_trail_base=b, stepped_trail_tight=t, stepped_trail_at=at,
        atr_period=14, label=f"{base.label}+{tag}")


def variants(base: Strategy) -> "list[Strategy]":
    out = []
    for tag, b, t, at in F_VARIANTS:
        out.append(replace(base,
                           id=f"{base.id}+{tag}",
                           exit=_exit_stepped(base.exit, b, t, at, tag)))
    return out
