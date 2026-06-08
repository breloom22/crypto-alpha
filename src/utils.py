"""Shared helpers: UTF-8 console setup, paths, small utilities."""
from __future__ import annotations

import os
import sys

# Project layout: this file lives in <root>/src/
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def setup_console() -> None:
    """Force UTF-8 stdout/stderr so box-drawing chars & emoji don't crash on
    Windows cp949 consoles. Safe to call repeatedly."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")        # py3.7+
        except Exception:                                # noqa: BLE001
            pass


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
