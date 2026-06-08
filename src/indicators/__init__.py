"""Indicator package: 35 leading-signal candidates across 5 categories."""
from .base import (
    Indicator, CrossAssetIndicator, build_indicators,
    ASSETS, SECTORS, CATEGORIES, clean_bool, market_index, market_frame,
)

__all__ = [
    "Indicator", "CrossAssetIndicator", "build_indicators",
    "ASSETS", "SECTORS", "CATEGORIES", "clean_bool", "market_index", "market_frame",
]
