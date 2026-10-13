"""
test_fetch.py
─────────────
The weekly price frame handed to the ranking must end on a complete week.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_prices as fp  # noqa: E402


def frame(last_row_priced: int, n: int = 10, weeks: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2026-09-11", periods=weeks, freq="W-FRI")
    px = pd.DataFrame(np.full((weeks, n), 10.0), index=idx, columns=[f"T{i}" for i in range(n)])
    px.iloc[-1, last_row_priced:] = np.nan
    return px


def test_drops_a_latest_bar_most_names_lack():
    # The 2026-10-13 failure: a new-week bar priced for 1 of the names.
    px = frame(last_row_priced=1)
    w, dv, dropped = fp.trim_incomplete_tail(px, px.copy())
    assert dropped == [px.index[-1]]
    assert w.index[-1] == px.index[-2] and len(dv) == len(w)
    assert w.iloc[-1].notna().all()


def test_keeps_a_latest_bar_with_normal_gaps():
    # A few halted or delisted names missing is normal and must not drop the week.
    px = frame(last_row_priced=8)
    w, _, dropped = fp.trim_incomplete_tail(px, px.copy())
    assert dropped == [] and len(w) == len(px)


def test_drops_consecutive_incomplete_bars():
    px = frame(last_row_priced=10)
    px.iloc[-2:, 2:] = np.nan
    w, _, dropped = fp.trim_incomplete_tail(px, px.copy())
    assert len(dropped) == 2 and w.index[-1] == px.index[-3]
