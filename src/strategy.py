"""
strategy.py — Strategy model + combinatorial generator (Part 4).

A Strategy binds an entry (one signal, an AND-combo, or a SEQ long/short pair)
to a direction, an :class:`ExitConfig`, and an optional list of filters. It
knows how to materialise its per-asset entry series (long/short), which the
backtester then turns into trades.

Layers (per the spec):
  L1  every (signal, direction) with the base exit
  L2  the L1 set with each of the 5 exit variants
  L3  filters AND-ed onto the IS-top strategies            (post-IS)
  L4  pairwise AND combos of IS-top strategies             (post-IS)
  L5  SEQ pairs: a top SHORT signal + a top LONG signal     (post-IS)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from exits import ExitConfig, EXIT_VARIANTS, LAYER2_VARIANTS
from filters import apply_filters
from signals import registry

_DIR_TAG = {"LONG": "L", "SHORT": "S"}


@dataclass
class Strategy:
    id: str
    members: tuple                 # tuple[(signal_id, direction), ...]
    direction: str                 # primary direction ("LONG"/"SHORT"/"BOTH" for SEQ)
    exit: ExitConfig
    filters: tuple = ()            # filter ids, e.g. ("F1",)
    combo_type: "str | None" = None  # None | "AND" | "SEQ"
    group: str = ""                # signal group of the primary member (for reporting)
    signal_params: dict = field(default_factory=dict)  # param overrides (sensitivity)

    # -- entry materialisation ----------------------------------------------
    def build_entries(self, data: "dict[str, pd.DataFrame]", symbol: str):
        """Return (long_entries, short_entries, opp_long, opp_short) boolean
        Series aligned to data[symbol].index. opp_* feed the E2 opposite-signal
        exit (raw, unfiltered). Unused series are all-False."""
        idx = data[symbol].index
        false = pd.Series(False, index=idx)
        long_e = false.copy()
        short_e = false.copy()

        if self.combo_type == "AND":
            d = self.direction
            acc = None
            for sid, _ in self.members:
                e = registry.compute_entries(registry.get(sid), data, symbol, d, self.signal_params or None)
                acc = e if acc is None else (acc & e)
            target = long_e if d == "LONG" else short_e
            target.loc[:] = acc.fillna(False)
        elif self.combo_type == "SEQ":
            for sid, d in self.members:
                e = registry.compute_entries(registry.get(sid), data, symbol, d, self.signal_params or None)
                if d == "LONG":
                    long_e = long_e | e
                else:
                    short_e = short_e | e
        else:  # single
            sid, d = self.members[0]
            e = registry.compute_entries(registry.get(sid), data, symbol, d, self.signal_params or None)
            if d == "LONG":
                long_e = e
            else:
                short_e = e

        # apply filters to the populated entry side(s)
        if self.filters:
            if long_e.any():
                long_e = apply_filters(long_e, self.filters, data, symbol, "LONG")
            if short_e.any():
                short_e = apply_filters(short_e, self.filters, data, symbol, "SHORT")

        # opposite-signal series for E2 (only for single-direction strategies)
        opp_long = false.copy()
        opp_short = false.copy()
        if self.exit.opposite_signal and self.combo_type is None:
            sid, d = self.members[0]
            spec = registry.get(sid)
            opp_dir = "SHORT" if d == "LONG" else "LONG"
            if opp_dir in spec.directions:
                opp = registry.compute_entries(spec, data, symbol, opp_dir, self.signal_params or None)
                if opp_dir == "LONG":
                    opp_long = opp
                else:
                    opp_short = opp
        return long_e.fillna(False), short_e.fillna(False), opp_long, opp_short

    @property
    def signal_ids(self) -> tuple:
        return tuple(sid for sid, _ in self.members)


# ---------------------------------------------------------------------------
def _single_id(sid: str, direction: str, exit_label: str, filters=()) -> str:
    base = f"{sid}_{_DIR_TAG[direction]}_{exit_label}"
    if filters:
        base += "_" + "".join(filters)
    return base


class StrategyGenerator:
    """Generates strategies layer by layer. Post-IS layers take already-ranked
    Strategy lists."""

    def __init__(self, specs: "list[registry.SignalSpec]"):
        self.specs = specs
        self._by_id = {s.id: s for s in specs}

    # -- L1: every (signal, direction) with the base exit -------------------
    def generate_layer1(self) -> "list[Strategy]":
        out = []
        for spec in self.specs:
            for d in spec.directions:
                out.append(Strategy(
                    id=_single_id(spec.id, d, "base"),
                    members=((spec.id, d),), direction=d,
                    exit=EXIT_VARIANTS["base"], group=spec.group))
        return out

    # -- L2: exit variants on the L1 set ------------------------------------
    def generate_layer2(self, base_strategies: "list[Strategy] | None" = None,
                        variants=None) -> "list[Strategy]":
        base = base_strategies if base_strategies is not None else self.generate_layer1()
        variants = variants or LAYER2_VARIANTS
        out = []
        for s in base:
            if s.combo_type is not None:
                continue
            sid, d = s.members[0]
            for v in variants:
                out.append(Strategy(
                    id=_single_id(sid, d, v),
                    members=((sid, d),), direction=d,
                    exit=EXIT_VARIANTS[v], group=s.group))
        return out

    # -- L3: filters AND-ed onto IS-top strategies --------------------------
    def generate_layer3(self, top_strategies: "list[Strategy]", filter_ids,
                        top_n: int = 30) -> "list[Strategy]":
        out = []
        for s in top_strategies[:top_n]:
            if s.combo_type is not None:
                continue
            sid, d = s.members[0]
            for fid in filter_ids:
                # skip contradictory trend filters (LONG with downtrend filter)
                if d == "LONG" and fid == "F2":
                    continue
                if d == "SHORT" and fid == "F1":
                    continue
                new_filters = tuple(s.filters) + (fid,)
                out.append(Strategy(
                    id=_single_id(sid, d, s.exit.label, new_filters),
                    members=((sid, d),), direction=d, exit=s.exit,
                    filters=new_filters, group=s.group))
        return out

    # -- L4: pairwise AND combos of IS-top strategies -----------------------
    def generate_layer4(self, top_strategies: "list[Strategy]", k: int = 2,
                        top_n: int = 20) -> "list[Strategy]":
        singles = [s for s in top_strategies[:top_n] if s.combo_type is None]
        out = []
        seen = set()
        for combo in itertools.combinations(singles, k):
            dirs = {s.direction for s in combo}
            if len(dirs) != 1:           # AND only makes sense same-direction
                continue
            d = dirs.pop()
            members = tuple((s.members[0][0], d) for s in combo)
            key = (tuple(sorted(m[0] for m in members)), d)
            if key in seen:
                continue
            seen.add(key)
            cid = "+".join(_single_id(sid, d, "and") for sid, _ in members)
            out.append(Strategy(
                id=cid, members=members, direction=d,
                exit=EXIT_VARIANTS["base"], combo_type="AND",
                group=self._by_id[members[0][0]].group))
        return out

    # -- L5: SEQ pairs (top SHORT signal + top LONG signal) -----------------
    def generate_layer5(self, top_short: "list[Strategy]",
                        top_long: "list[Strategy]", n: int = 5) -> "list[Strategy]":
        shorts = [s for s in top_short if s.direction == "SHORT"
                  and s.combo_type is None][:n]
        longs = [s for s in top_long if s.direction == "LONG"
                 and s.combo_type is None][:n]
        out = []
        for ss, ls in itertools.product(shorts, longs):
            ssid, lsid = ss.members[0][0], ls.members[0][0]
            members = ((ssid, "SHORT"), (lsid, "LONG"))
            cid = f"SEQ_{ssid}_S+{lsid}_L"
            out.append(Strategy(
                id=cid, members=members, direction="BOTH",
                exit=EXIT_VARIANTS["base"], combo_type="SEQ",
                group=self._by_id[ssid].group))
        return out
