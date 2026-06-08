"""
Improvement A — ATR-based dynamic SL/TP.

Replace a strategy's *fixed-percent* stop/target with one scaled to the entry
bar's ATR(14), so a volatile asset (DOGE) gets a wider stop than a calm one
(BTC). The signal logic and max-hold are untouched; only the stop/target
geometry changes. ATR is read at the entry bar inside the simulator
(look-ahead safe — see backtester_v3._simulate_one).

    A1 (standard)   sl 1.5  tp 3.0   1:2 R/R
    A2 (tight)      sl 1.0  tp 2.5   quick cut
    A3 (wide)       sl 2.0  tp 4.0   noise-tolerant
    A4 (asymmetric) sl 1.5  tp 5.0   fat-tail capture
"""
from __future__ import annotations

from dataclasses import replace

from strategy import Strategy
from exits import ExitConfig

# (label, sl_mult, tp_mult)
A_VARIANTS = (
    ("A1", 1.5, 3.0),
    ("A2", 1.0, 2.5),
    ("A3", 2.0, 4.0),
    ("A4", 1.5, 5.0),
)


def _exit_with_atr(base: ExitConfig, sl_mult: float, tp_mult: float, tag: str) -> ExitConfig:
    """Swap fixed SL/TP for ATR-multiple SL/TP, keep everything else (max_hold,
    any trailing/chandelier the base already had)."""
    return replace(
        base,
        stop_loss_pct=None, take_profit_pct=None,
        atr_stop_mult=sl_mult, atr_tp_mult=tp_mult, atr_period=14,
        label=f"{base.label}+{tag}")


def variants(base: Strategy) -> "list[Strategy]":
    """Return the A1..A4 versions of a base strategy."""
    out = []
    for tag, sl, tp in A_VARIANTS:
        out.append(replace(base,
                           id=f"{base.id}+{tag}",
                           exit=_exit_with_atr(base.exit, sl, tp, tag)))
    return out
