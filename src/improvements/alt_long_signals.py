"""
Improvement E — alternative LONG legs for the SEQ strategies.

The SEQ books are SHORT-heavy because their LONG signal (e.g. Supertrend flip)
fires rarely, defeating the two-sided intent. Swap in a more frequent / different
LONG signal while keeping the SHORT leg and the SEQ framework:

    E1  S4.04  Normalized-ATR overreaction reversal (already Tier-S proven)
    E2  S2.20  RSI(14) oversold cross-up (new, simplest mean-reversion long)
    E3  S2.01  Fisher Transform turn-up
    E4  S1.13  consecutive down-bars then an up-bar (price action)

Only SEQ strategies are affected; a variant that would reproduce the base's
existing LONG leg is skipped.
"""
from __future__ import annotations

from dataclasses import replace

from strategy import Strategy
from signals import registry
from signals.registry import LONG
from indicators import _ta


# --- new signal: RSI(14) oversold cross-up (LONG only) ---------------------
@registry.signal("S2.20", "RSI(14) oversold cross-up", "S2", (LONG,),
                 period=14, thresh=30)
def rsi_oversold_crossup(df, direction, period=14, thresh=30):
    """LONG: RSI(14) crosses from <=thresh back up through thresh (oversold
    bounce). Look-ahead safe (RSI uses trailing closes; cross uses t-1,t)."""
    r = _ta.rsi(df["close"], period)
    return _ta.crosses_above(r, thresh)


# (label, long_signal_id)
E_VARIANTS = (
    ("E1", "S4.04"),
    ("E2", "S2.20"),
    ("E3", "S2.01"),
    ("E4", "S1.13"),
)


def variants(base: Strategy) -> "list[Strategy]":
    if base.combo_type != "SEQ":
        return []
    short_member = base.members[0]                 # (short_sig, "SHORT")
    existing_long = base.members[1][0]
    out = []
    for tag, lsid in E_VARIANTS:
        if lsid == existing_long:
            continue                               # no-op for this base
        out.append(replace(base,
                           id=f"{base.id}+{tag}",
                           members=(short_member, (lsid, "LONG"))))
    return out
