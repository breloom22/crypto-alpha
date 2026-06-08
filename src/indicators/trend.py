"""Category B — Trend / structure indicators (T1..T7). Look-ahead safe.

Persistent "state" conditions are edge-triggered (fire when the state is first
entered) so each indicator emits discrete, comparable warning events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _ta
from .base import Indicator


def _edge(state: pd.Series) -> pd.Series:
    return (state & ~state.shift(1, fill_value=False)).fillna(False)


class T1_EMADeadCrossShort(Indicator):
    _id, _name, _category = "T1", "EMA Dead Cross (20/50)", "trend"
    _defaults = {"short": 20, "long": 50}

    def compute(self, df):
        return _ta.ema(df["close"], self.params["short"]) - _ta.ema(df["close"], self.params["long"])

    def generate_signals(self, df):
        es = _ta.ema(df["close"], self.params["short"])
        el = _ta.ema(df["close"], self.params["long"])
        return _ta.crosses_below(es, el)


class T2_MADeadCrossLong(Indicator):
    _id, _name, _category = "T2", "MA Dead Cross (50/200)", "trend"
    _defaults = {"short": 50, "long": 200}

    def compute(self, df):
        return _ta.sma(df["close"], self.params["short"]) - _ta.sma(df["close"], self.params["long"])

    def generate_signals(self, df):
        ss = _ta.sma(df["close"], self.params["short"])
        sl = _ta.sma(df["close"], self.params["long"])
        return _ta.crosses_below(ss, sl)


class T3_BelowEMA200(Indicator):
    _id, _name, _category = "T3", "Close Breaks Below EMA(200)", "trend"
    _defaults = {"period": 200}

    def compute(self, df):
        return df["close"] - _ta.ema(df["close"], self.params["period"])

    def generate_signals(self, df):
        e = _ta.ema(df["close"], self.params["period"])
        return _ta.crosses_below(df["close"], e)


class T4_ADXDirChange(Indicator):
    _id, _name, _category = "T4", "ADX +DI/-DI Bearish Cross", "trend"
    _defaults = {"period": 14, "adx_threshold": 25}

    def compute(self, df):
        plus_di, minus_di, _ = _ta.adx(df["high"], df["low"], df["close"], self.params["period"])
        return plus_di - minus_di

    def generate_signals(self, df):
        plus_di, minus_di, adx_v = _ta.adx(df["high"], df["low"], df["close"], self.params["period"])
        return _ta.crosses_below(plus_di, minus_di) & (adx_v > self.params["adx_threshold"])


class T5_ParabolicSARFlip(Indicator):
    _id, _name, _category = "T5", "Parabolic SAR Bearish Flip", "trend"
    _defaults = {"af": 0.02, "max_af": 0.2}

    def compute(self, df):
        return _ta.parabolic_sar(df["high"], df["low"], self.params["af"], self.params["max_af"])

    def generate_signals(self, df):
        # use the engine's actual uptrend->downtrend transition (low pierces SAR)
        return _ta.parabolic_sar_bear_flip(df["high"], df["low"],
                                           self.params["af"], self.params["max_af"])


class T6_LowerHighLowerLow(Indicator):
    _id, _name, _category = "T6", "Lower High + Lower Low", "trend"
    _defaults = {"lookback": 20}

    def compute(self, df):
        n = self.params["lookback"]
        return df["high"].rolling(n, min_periods=n).max() - df["high"].rolling(n, min_periods=n).max().shift(n)

    def generate_signals(self, df):
        n = self.params["lookback"]
        recent_high = df["high"].rolling(n, min_periods=n).max()
        recent_low = df["low"].rolling(n, min_periods=n).min()
        prev_high = recent_high.shift(n)
        prev_low = recent_low.shift(n)
        state = (recent_high < prev_high) & (recent_low < prev_low)
        return _edge(state.fillna(False))


class T7_IchimokuBreakdown(Indicator):
    _id, _name, _category = "T7", "Ichimoku Cloud Breakdown", "trend"
    _defaults = {"tenkan": 9, "kijun": 26, "senkou": 52}

    def compute(self, df):
        _, _, span_a, span_b = _ta.ichimoku(df["high"], df["low"],
                                            self.params["tenkan"], self.params["kijun"], self.params["senkou"])
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
        return df["close"] - cloud_bottom

    def generate_signals(self, df):
        _, _, span_a, span_b = _ta.ichimoku(df["high"], df["low"],
                                            self.params["tenkan"], self.params["kijun"], self.params["senkou"])
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
        return _ta.crosses_below(df["close"], cloud_bottom)


def get_indicators():
    return [T1_EMADeadCrossShort(), T2_MADeadCrossLong(), T3_BelowEMA200(),
            T4_ADXDirChange(), T5_ParabolicSARFlip(), T6_LowerHighLowerLow(),
            T7_IchimokuBreakdown()]
