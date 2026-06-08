"""Category A — Momentum indicators (M1..M8). All look-ahead safe."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _ta
from .base import Indicator


def _edge(state: pd.Series) -> pd.Series:
    return (state & ~state.shift(1, fill_value=False)).fillna(False)


class M1_RSIOverbought(Indicator):
    _id, _name, _category = "M1", "RSI Overbought Exit", "momentum"
    _defaults = {"period": 14, "threshold": 70}

    def compute(self, df):
        return _ta.rsi(df["close"], self.params["period"])

    def generate_signals(self, df):
        r = self.compute(df)
        thr = self.params["threshold"]
        # was overbought (>thr) recently, now drops back below thr
        return _ta.crosses_below(r, thr) & (r.shift(1) > thr)


class M2_RSIBearishDivergence(Indicator):
    _id, _name, _category = "M2", "RSI Bearish Divergence", "momentum"
    _defaults = {"period": 14, "lookback": 20}

    def compute(self, df):
        return _ta.rsi(df["close"], self.params["period"])

    def generate_signals(self, df):
        n = self.params["lookback"]
        rsi = self.compute(df)
        close = df["close"]
        # price makes a higher high vs the prior window while RSI makes a lower
        # high -> bearish divergence. Compare current rolling-high to the high
        # of the PREVIOUS window (shifted) to avoid trivial equality.
        price_hh = close == close.rolling(n, min_periods=n).max()
        prev_price_high = close.rolling(n, min_periods=n).max().shift(n)
        prev_rsi_high = rsi.rolling(n, min_periods=n).max().shift(n)
        higher_price = close > prev_price_high
        lower_rsi = rsi < prev_rsi_high
        # edge-trigger: one signal per divergence onset (a divergence condition
        # can hold for several consecutive bars), matching every other indicator.
        return _edge((price_hh & higher_price & lower_rsi).fillna(False))


class M3_MACDDeadCross(Indicator):
    _id, _name, _category = "M3", "MACD Dead Cross", "momentum"
    _defaults = {"fast": 12, "slow": 26, "signal": 9}

    def compute(self, df):
        line, sig, _ = _ta.macd(df["close"], self.params["fast"],
                                self.params["slow"], self.params["signal"])
        return line - sig

    def generate_signals(self, df):
        line, sig, _ = _ta.macd(df["close"], self.params["fast"],
                                self.params["slow"], self.params["signal"])
        return _ta.crosses_below(line, sig)


class M4_MACDHistFlip(Indicator):
    _id, _name, _category = "M4", "MACD Histogram Neg Flip", "momentum"
    _defaults = {"fast": 12, "slow": 26, "signal": 9}

    def compute(self, df):
        _, _, hist = _ta.macd(df["close"], self.params["fast"],
                              self.params["slow"], self.params["signal"])
        return hist

    def generate_signals(self, df):
        hist = self.compute(df)
        return ((hist < 0) & (hist.shift(1) >= 0)).fillna(False)


class M5_StochCross(Indicator):
    _id, _name, _category = "M5", "Stochastic %K/%D Cross (>80)", "momentum"
    _defaults = {"k": 14, "d": 3, "threshold": 80}

    def compute(self, df):
        k, d = _ta.stochastic(df["high"], df["low"], df["close"],
                              self.params["k"], self.params["d"])
        return k - d

    def generate_signals(self, df):
        k, d = _ta.stochastic(df["high"], df["low"], df["close"],
                              self.params["k"], self.params["d"])
        thr = self.params["threshold"]
        return _ta.crosses_below(k, d) & (k.shift(1) > thr)


class M6_CCIReversal(Indicator):
    _id, _name, _category = "M6", "CCI Extreme Reversal", "momentum"
    _defaults = {"period": 20, "threshold": 100}

    def compute(self, df):
        return _ta.cci(df["high"], df["low"], df["close"], self.params["period"])

    def generate_signals(self, df):
        c = self.compute(df)
        thr = self.params["threshold"]
        return _ta.crosses_below(c, thr) & (c.shift(1) > thr)


class M7_WilliamsR(Indicator):
    _id, _name, _category = "M7", "Williams %R Overbought Exit", "momentum"
    _defaults = {"period": 14, "threshold": -20}

    def compute(self, df):
        return _ta.williams_r(df["high"], df["low"], df["close"], self.params["period"])

    def generate_signals(self, df):
        w = self.compute(df)
        thr = self.params["threshold"]
        return _ta.crosses_below(w, thr) & (w.shift(1) > thr)


class M8_ROCTurn(Indicator):
    _id, _name, _category = "M8", "ROC Negative Turn", "momentum"
    _defaults = {"period": 10}

    def compute(self, df):
        return _ta.roc(df["close"], self.params["period"])

    def generate_signals(self, df):
        r = self.compute(df)
        # ROC turns from positive to negative (momentum rolling over)
        return ((r < 0) & (r.shift(1) >= 0)).fillna(False)


def get_indicators():
    return [M1_RSIOverbought(), M2_RSIBearishDivergence(), M3_MACDDeadCross(),
            M4_MACDHistFlip(), M5_StochCross(), M6_CCIReversal(),
            M7_WilliamsR(), M8_ROCTurn()]
