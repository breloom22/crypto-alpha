"""
reconstruct.py — rebuild the 7 Phase-4 *base* strategies from their v3 IDs.

The v3 generator names strategies deterministically (strategy.py):

    single : f"{sid}_{DIR}_{exit_label}" [+ "".join(filters)]
    SEQ    : f"SEQ_{shortSig}_S+{longSig}_L"   (exit = base)

so every target ID maps back to a unique (members, direction, exit, filters)
tuple. We reconstruct the exact :class:`Strategy` objects v3 evaluated rather
than re-running the whole discovery pipeline — same objects, same numbers.
"""
from __future__ import annotations

import re

from strategy import Strategy
from exits import EXIT_VARIANTS
from signals import registry

_DIR_TAG = {"L": "LONG", "S": "SHORT"}

# The seven Phase-4 targets (PASSED_STRATEGIES.md §2 / v4 spec §대상 전략).
TARGET_IDS = (
    "SEQ_S1.11_S+S4.04_L",
    "S2.08_L_aggr_F9",
    "S2.08_L_aggr_F3",
    "S3.12_S_aggr_F8",
    "S1.11_S_aggr",
    "SEQ_S1.11_S+S3.01_L",
    "S7.04_S_trail",
)

# Tier labels from the report, for display only.
TARGET_TIER = {
    "SEQ_S1.11_S+S4.04_L": "S",
    "S2.08_L_aggr_F9": "A",
    "S2.08_L_aggr_F3": "A",
    "S3.12_S_aggr_F8": "B",
    "S1.11_S_aggr": "B",
    "SEQ_S1.11_S+S3.01_L": "B",
    "S7.04_S_trail": "B",
}

# Original OOS return (%) from PASSED_STRATEGIES.md, for a quick reconstruction
# sanity check (not used as ground truth — the live backtest is authoritative).
REPORT_OOS_RETURN = {
    "SEQ_S1.11_S+S4.04_L": 45.0,
    "S2.08_L_aggr_F9": 16.6,
    "S2.08_L_aggr_F3": 11.8,
    "S3.12_S_aggr_F8": 43.6,
    "S1.11_S_aggr": 24.1,
    "SEQ_S1.11_S+S3.01_L": 38.9,
    "S7.04_S_trail": 14.8,
}

_SEQ_RE = re.compile(r"^SEQ_(?P<short>S\d+\.\d+)_S\+(?P<long>S\d+\.\d+)_L$")
_SINGLE_RE = re.compile(
    r"^(?P<sid>S\d+\.\d+)_(?P<dir>[LS])_(?P<exit>base|cons|aggr|trail|quick|atr)"
    r"(?:_(?P<filters>(?:F\d+)+))?$")


def _group_of(sid: str) -> str:
    return sid.split(".")[0]


def reconstruct(strategy_id: str) -> Strategy:
    """Rebuild the v3 :class:`Strategy` for a target ID (single or SEQ)."""
    registry.load_all()

    m = _SEQ_RE.match(strategy_id)
    if m:
        ssid, lsid = m.group("short"), m.group("long")
        return Strategy(
            id=strategy_id,
            members=((ssid, "SHORT"), (lsid, "LONG")),
            direction="BOTH",
            exit=EXIT_VARIANTS["base"],
            combo_type="SEQ",
            group=_group_of(ssid))

    m = _SINGLE_RE.match(strategy_id)
    if m:
        sid = m.group("sid")
        direction = _DIR_TAG[m.group("dir")]
        exit_label = m.group("exit")
        filters = tuple(re.findall(r"F\d+", m.group("filters") or ""))
        return Strategy(
            id=strategy_id,
            members=((sid, direction),),
            direction=direction,
            exit=EXIT_VARIANTS[exit_label],
            filters=filters,
            group=_group_of(sid))

    raise ValueError(f"Cannot parse strategy id: {strategy_id!r}")


def reconstruct_targets() -> "list[Strategy]":
    """The 7 Phase-4 base strategies, in report order."""
    return [reconstruct(sid) for sid in TARGET_IDS]
