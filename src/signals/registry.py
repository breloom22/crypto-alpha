"""
registry.py — signal contract + global registry for the v3 strategy engine.

A *signal* answers one question for one (asset, direction): on which days does
an entry trigger? The convention is identical to the rest of the pipeline:

    a signal that is True at day t  =>  enter at the NEXT day's open (t+1).

Two flavours share one registry so the backtester treats them uniformly:

* per-asset signal:
      def fn(df: pd.DataFrame, direction: str, **params) -> pd.Series[bool]
  `df` is one asset's OHLCV frame; returns a boolean Series aligned to df.index.

* cross-asset signal (group S7):
      def fn(data: dict[str, pd.DataFrame], symbol: str, direction: str,
             **params) -> pd.Series[bool]
  `data` is the full {symbol: DataFrame} universe; the function returns the
  entry series for `symbol` (it may read every asset to do so).

`direction` is "LONG" or "SHORT". Every signal must be LOOK-AHEAD SAFE: the
value at t may depend only on data at indices <= t. Entry at t+1 open is what
makes the t-th signal tradeable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

LONG = "LONG"
SHORT = "SHORT"
BOTH = ("LONG", "SHORT")

GROUP_NAMES = {
    "S1": "Price Action / Candles",
    "S2": "Alt Momentum",
    "S3": "Alt Trend",
    "S4": "Volatility / Range",
    "S5": "Alt Volume",
    "S6": "Statistical",
    "S7": "Cross-Asset / Relative Strength",
    "S8": "Non-standard Reuse",
}


@dataclass(frozen=True)
class SignalSpec:
    id: str                       # e.g. "S2.03"
    name: str                     # human label
    group: str                    # "S1".."S8"
    fn: Callable                  # the signal function (see module docstring)
    directions: tuple             # subset of ("LONG", "SHORT")
    cross_asset: bool = False
    params: dict = field(default_factory=dict)


# Global registry (id -> SignalSpec). Last registration of an id wins, so
# re-importing a module during interactive work is harmless.
REGISTRY: "dict[str, SignalSpec]" = {}


def register(spec: SignalSpec) -> SignalSpec:
    REGISTRY[spec.id] = spec
    return spec


def signal(id: str, name: str, group: str, directions=BOTH,
           cross_asset: bool = False, **params):
    """Decorator: register a signal function and return it unchanged.

    Usage:
        @signal("S2.03", "Connors RSI", "S2", BOTH, rsi_period=3)
        def connors(df, direction, rsi_period=3, ...):
            ...
    """
    def deco(fn: Callable) -> Callable:
        register(SignalSpec(id=id, name=name, group=group, fn=fn,
                            directions=tuple(directions), cross_asset=cross_asset,
                            params=dict(params)))
        return fn
    return deco


def get(signal_id: str) -> SignalSpec:
    return REGISTRY[signal_id]


def all_specs() -> "list[SignalSpec]":
    return list(REGISTRY.values())


def compute_entries(spec: SignalSpec, data: "dict[str, pd.DataFrame]",
                    symbol: str, direction: str,
                    params: "dict | None" = None) -> pd.Series:
    """Dispatch a signal to a clean boolean entry Series for (symbol, direction).

    Handles both per-asset and cross-asset signals and guarantees the result is
    a boolean Series aligned to data[symbol].index with NaN -> False.
    """
    df = data[symbol]
    p = {**spec.params, **(params or {})}
    if spec.cross_asset:
        raw = spec.fn(data, symbol, direction, **p)
    else:
        raw = spec.fn(df, direction, **p)
    return pd.Series(raw).reindex(df.index).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
def load_all() -> "dict[str, SignalSpec]":
    """Import every signal module so the registry is fully populated, then
    return it. Modules register their signals at import time."""
    from . import (price_action, alt_momentum, alt_trend, volatility,
                   alt_volume, statistical, cross_asset, nonstandard)  # noqa: F401
    return REGISTRY
