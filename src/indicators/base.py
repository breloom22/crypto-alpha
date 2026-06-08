"""
base.py — uniform indicator contract + registry.

Two flavours of indicator share one interface so the backtester can treat them
identically:

* Indicator            — per-asset. compute()/generate_signals() take a single
                         OHLCV DataFrame.
* CrossAssetIndicator  — market-level (X1..X7). compute()/generate_signals()
                         take the full {symbol: DataFrame} dict and return ONE
                         market-wide series (evaluated against every asset's
                         crashes by the backtester).

Every signal series returned to callers is a clean boolean Series indexed by
date with NaN -> False. All indicators must be look-ahead safe.
"""
from __future__ import annotations

from collections import OrderedDict

import pandas as pd

# Asset universe & sector map (shared by cross-asset indicators) -------------
ASSETS = ["BTC", "ETH", "SOL", "DOGE", "OP", "AVAX", "XRP", "XLM", "SUI"]
SECTORS = {
    "L1":    ["BTC", "ETH", "SOL", "AVAX", "SUI"],
    "Meme":  ["DOGE"],
    "Infra": ["OP", "XRP", "XLM"],
}
CATEGORIES = {
    "momentum": "Momentum", "trend": "Trend/Structure",
    "volatility": "Volatility", "volume": "Volume", "cross_asset": "Cross-Asset",
}


class Indicator:
    """Per-asset indicator base class."""
    cross_asset = False
    _id = "BASE"
    _name = "base"
    _category = "momentum"
    _defaults: dict = {}

    def __init__(self, **params):
        self.id = self._id
        self.name = self._name
        self.category = self._category
        self.params = {**self._defaults, **params}

    # -- to be overridden ----------------------------------------------------
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Return the underlying float series (for inspection/plots)."""
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a RAW boolean Series (True = warning fired)."""
        raise NotImplementedError

    # -- public, cleaned -----------------------------------------------------
    def signals(self, df: pd.DataFrame) -> pd.Series:
        sig = self.generate_signals(df)
        return clean_bool(sig, df.index)

    def with_params(self, **overrides) -> "Indicator":
        return type(self)(**{**self.params, **overrides})

    def __repr__(self):
        return f"<{self.id} {self.name} {self.params}>"


class CrossAssetIndicator(Indicator):
    """Market-level indicator. Operates on the full data dict."""
    cross_asset = True
    _category = "cross_asset"

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.Series:   # type: ignore[override]
        raise NotImplementedError

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.Series:  # type: ignore[override]
        raise NotImplementedError

    def signals(self, data: dict[str, pd.DataFrame]) -> pd.Series:   # type: ignore[override]
        sig = self.generate_signals(data)
        return clean_bool(sig, sig.index)


# ---- helpers ---------------------------------------------------------------
def clean_bool(sig: pd.Series, index) -> pd.Series:
    """Reindex to `index`, fill NaN with False, coerce to bool."""
    return sig.reindex(index).fillna(False).astype(bool)


def market_index(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Sorted union of all assets' dates (the market calendar)."""
    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.union(df.index)
    return idx.sort_values()


def market_frame(data: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    """Wide DataFrame (date x symbol) of one OHLCV field across assets."""
    return pd.DataFrame({s: df[field] for s, df in data.items()}).sort_index()


# ---- registry --------------------------------------------------------------
def build_indicators() -> "OrderedDict[str, Indicator]":
    """Instantiate all 35 indicators (default params), keyed by id in order."""
    from . import momentum, trend, volatility, volume, cross_asset
    reg: "OrderedDict[str, Indicator]" = OrderedDict()
    for mod in (momentum, trend, volatility, volume, cross_asset):
        for ind in mod.get_indicators():
            if ind.id in reg:
                raise ValueError(f"duplicate indicator id {ind.id}")
            reg[ind.id] = ind
    return reg
