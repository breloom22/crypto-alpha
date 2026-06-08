"""
exits.py — Exit Rule Library (E1–E12) for the v3 strategy engine.

An exit is fully described by an :class:`ExitConfig`. The per-trade simulator in
``backtester_v3`` interprets the config; keeping exits declarative (rather than a
bag of callables) lets the simulator enforce the spec's priority rules
correctly — notably "SL has priority over TP on the same bar" and "earliest
trigger wins" across several simultaneously-active stop/target mechanisms.

Percentage fields are *fractions* relative to the entry fill (sl=-0.05 = -5%).
For SHORT trades the simulator mirrors the geometry automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class ExitConfig:
    # E1 — fixed max holding period (bars)
    max_hold: "int | None" = None
    # E2 — exit on opposite-direction signal of the same strategy
    opposite_signal: bool = False
    # E3 / E4 — fixed percent stop-loss / take-profit (fractions, sl<0, tp>0)
    stop_loss_pct: "float | None" = None
    take_profit_pct: "float | None" = None
    # E5 — trailing stop: exit when price retraces this fraction from the peak
    trailing_pct: "float | None" = None
    # E6 — indicator neutral return (RSI back to 50). Simulator uses RSI(14).
    indicator_neutral: bool = False
    # E7 — volatility-proportional max hold = max(7, 2*atr_period)
    time_dynamic: bool = False
    # E8 — ATR-based stop / target (multiples of ATR(atr_period) at entry)
    atr_stop_mult: "float | None" = None
    atr_tp_mult: "float | None" = None
    atr_period: int = 14
    # E9 — Chandelier exit (trailing extreme ± mult*ATR(period))
    chandelier_mult: "float | None" = None
    chandelier_period: int = 22
    # E10 — profit target then trailing: after +trigger, trail by mult*ATR
    profit_trail_trigger: "float | None" = None
    profit_trail_atr: "float | None" = None
    # E11 — break-even stop: after +trigger, move stop to entry
    breakeven_trigger: "float | None" = None
    # E12 — exit on the first profitable close
    first_profit_close: bool = False
    # --- Phase 4 (v4) additions (default-None => identical to v3 when unset) ---
    # E13 — two-tier stepped ATR trailing: trail the peak/trough by
    #   stepped_trail_base * ATR, tightening to stepped_trail_tight * ATR once
    #   unrealised profit reaches stepped_trail_at. Uses ATR(atr_period) @ entry.
    stepped_trail_base: "float | None" = None
    stepped_trail_tight: "float | None" = None
    stepped_trail_at: "float | None" = None
    # FC3 — loss-streak guard: skip ONE entry signal whenever the last
    #   `skip_after_n_losses` *closed* trades of this (strategy, asset) were all
    #   losers (an execution rule, hence it lives on the exit config).
    skip_after_n_losses: "int | None" = None
    # v5 — per-strategy cooldown override (bars after an exit before re-entry).
    #   None => use the backtester's call-level cooldown (so v3/v4 unchanged).
    cooldown: "int | None" = None
    # display label (used in strategy ids)
    label: str = "base"


# ---------------------------------------------------------------------------
# E1..E12 fragment constructors (compose with ``compose``)
# ---------------------------------------------------------------------------
def E1(max_hold: int) -> ExitConfig:
    return ExitConfig(max_hold=max_hold)


def E2() -> ExitConfig:
    return ExitConfig(opposite_signal=True)


def E3(stop_loss_pct: float) -> ExitConfig:
    return ExitConfig(stop_loss_pct=stop_loss_pct)


def E4(take_profit_pct: float) -> ExitConfig:
    return ExitConfig(take_profit_pct=take_profit_pct)


def E5(trailing_pct: float) -> ExitConfig:
    return ExitConfig(trailing_pct=trailing_pct)


def E6() -> ExitConfig:
    return ExitConfig(indicator_neutral=True)


def E7() -> ExitConfig:
    return ExitConfig(time_dynamic=True)


def E8(atr_stop_mult: "float | None" = None, atr_tp_mult: "float | None" = None,
       atr_period: int = 14) -> ExitConfig:
    return ExitConfig(atr_stop_mult=atr_stop_mult, atr_tp_mult=atr_tp_mult,
                      atr_period=atr_period)


def E9(chandelier_mult: float = 3.0, chandelier_period: int = 22) -> ExitConfig:
    return ExitConfig(chandelier_mult=chandelier_mult, chandelier_period=chandelier_period)


def E10(trigger: float = 0.03, atr_mult: float = 1.5) -> ExitConfig:
    return ExitConfig(profit_trail_trigger=trigger, profit_trail_atr=atr_mult)


def E11(trigger: float = 0.02) -> ExitConfig:
    return ExitConfig(breakeven_trigger=trigger)


def E12() -> ExitConfig:
    return ExitConfig(first_profit_close=True)


def E13(base_mult: float = 2.0, tight_mult: float = 1.0, tighten_at: float = 0.05,
        atr_period: int = 14) -> ExitConfig:
    """Two-tier stepped ATR trailing (Improvement F)."""
    return ExitConfig(stepped_trail_base=base_mult, stepped_trail_tight=tight_mult,
                      stepped_trail_at=tighten_at, atr_period=atr_period)


_DEFAULTS = ExitConfig()


def compose(*configs: ExitConfig, label: str = "base") -> ExitConfig:
    """Merge several exit fragments into one config. For each field the last
    non-default value wins; ``atr_period`` etc. carry through."""
    merged = {}
    for cfg in configs:
        for f in fields(ExitConfig):
            if f.name == "label":
                continue
            val = getattr(cfg, f.name)
            if val != getattr(_DEFAULTS, f.name):
                merged[f.name] = val
    return replace(_DEFAULTS, label=label, **merged)


# ---------------------------------------------------------------------------
# Exit variants used by the strategy generator (Part 4, Layer 1 & 2)
# Short labels feed the strategy-id naming scheme.
# ---------------------------------------------------------------------------
EXIT_VARIANTS = {
    # Layer 1 default: -5% SL, +10% TP, 10-day max hold
    "base":  compose(E3(-0.05), E4(0.10), E1(10), label="base"),
    # Layer 2 variants
    "cons":  compose(E3(-0.03), E4(0.05), E1(7),  label="cons"),    # Conservative
    "aggr":  compose(E3(-0.07), E4(0.15), E1(14), label="aggr"),    # Aggressive
    "trail": compose(E5(0.03), E1(14), label="trail"),              # Trailing stop
    "quick": compose(E12(), E3(-0.03), E1(5), label="quick"),       # Quick scalp
    "atr":   compose(E8(atr_stop_mult=2.0, atr_tp_mult=3.0, atr_period=14),
                     E1(10), label="atr"),                          # ATR-based
}

# Variants applied in Layer 2 (everything except the base, which is Layer 1).
LAYER2_VARIANTS = ["cons", "aggr", "trail", "quick", "atr"]
