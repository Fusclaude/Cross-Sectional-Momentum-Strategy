"""
test_breakout.py
────────────────
Rule tests for the all-time-high breakout scanner, on hand-built price paths
where the right answer is obvious, plus the same point-in-time truncation
check the factor engine gets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import breakout as bo  # noqa: E402

BCFG = {"min_history_weeks": 10, "min_base_weeks": 4, "recent_weeks": 3,
        "stop_pct": 0.08, "max_extension": 0.15, "trailing_stop": 0.25,
        "max_positions": 5, "volume_lookback_weeks": 20}


def frame(*cols: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-05", periods=len(cols[0]), freq="W-FRI")
    return pd.DataFrame({f"S{i}": c for i, c in enumerate(cols)}, index=idx)


def base_then(tail: list[float]) -> list[float]:
    """Rise to a high of 100 at week 5, drift in a 90-99 range, then `tail`."""
    return [60, 70, 80, 90, 95, 100] + [92, 95, 90, 97, 94, 99, 96, 98] + tail


def test_breakout_detected_after_base():
    px = frame(base_then([103, 104]))
    st = bo.breakout_state(px, BCFG)
    i = 14  # the 103 close
    assert st["weeks_since_breakout"]["S0"].iloc[i] == 0
    assert st["breakout_level"]["S0"].iloc[i] == 100
    assert st["base_weeks"]["S0"].iloc[i] == 9
    assert st["candidate"]["S0"].iloc[i] == 1
    assert st["candidate"]["S0"].iloc[i + 1] == 1
    assert st["weeks_since_breakout"]["S0"].iloc[i + 1] == 1   # 104 is continuation
    assert st["candidate"]["S0"].iloc[:i].sum() == 0


def test_no_breakout_without_enough_history():
    px = frame(base_then([103]))
    st = bo.breakout_state(px, {**BCFG, "min_history_weeks": 30})
    assert st["candidate"]["S0"].sum() == 0


def test_continuation_is_not_a_breakout():
    """New high every week: the prior high is always 1 week old, never a base."""
    px = frame(list(np.linspace(50, 150, 30)))
    st = bo.breakout_state(px, BCFG)
    assert st["weeks_since_breakout"]["S0"].isna().all()
    assert st["candidate"]["S0"].sum() == 0
    assert (st["pct_from_ath"]["S0"] == 0).all()


def test_failed_breakout_stays_dead_until_new_breakout():
    # break out to 103, then close below 100 * 0.92, then recover to 101
    px = frame(base_then([103, 91, 101]))
    st = bo.breakout_state(px, BCFG)
    assert st["failed"]["S0"].iloc[15] == 1
    assert st["candidate"]["S0"].iloc[15] == 0
    assert st["candidate"]["S0"].iloc[16] == 0   # recovered, but still failed


def test_overextended_is_not_a_candidate():
    px = frame(base_then([120]))
    st = bo.breakout_state(px, BCFG)
    assert st["weeks_since_breakout"]["S0"].iloc[-1] == 0
    assert st["candidate"]["S0"].iloc[-1] == 0


def test_candidate_expires_after_recent_weeks():
    px = frame(base_then([103, 103, 103, 103, 103]))
    st = bo.breakout_state(px, BCFG)
    assert list(st["candidate"]["S0"].iloc[14:]) == [1, 1, 1, 1, 0]


def test_min_price_screen():
    px = frame(base_then([103]))
    st = bo.breakout_state(px, BCFG, min_price=200)
    assert st["candidate"]["S0"].sum() == 0


def test_gaps_do_not_break_state():
    s = base_then([103, 104])
    s[8] = np.nan
    st = bo.breakout_state(frame(s), BCFG)
    assert st["candidate"]["S0"].iloc[14] == 1
    assert np.isnan(st["ath"]["S0"].iloc[8])


@pytest.fixture(scope="module")
def random_prices():
    rng = np.random.default_rng(7)
    T, N = 260, 40
    rets = rng.normal(0.002, 0.04, (T, N))
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range("2020-01-03", periods=T, freq="W-FRI")
    return pd.DataFrame(px, index=idx, columns=[f"T{i:02d}" for i in range(N)])


def test_point_in_time_truncation(random_prices):
    """Rebuild from truncated history: no historical value may change."""
    full = bo.breakout_state(random_prices, BCFG)
    for cut in [80, 150, 220]:
        part = bo.breakout_state(random_prices.iloc[:cut], BCFG)
        for k in bo.STATE_KEYS:
            pd.testing.assert_frame_equal(part[k], full[k].iloc[:cut], check_freq=False)


def test_backtest_and_event_study_run(random_prices):
    st = bo.breakout_state(random_prices, BCFG)
    rs = bo.rs_rating(random_prices, 26)
    es = bo.event_study(st, random_prices, [4, 13])
    assert es["n_events"] > 0 and "4" in es["horizons"]
    bt = bo.backtest(st, random_prices, BCFG, rs)
    assert "error" not in bt
    assert 0 < bt["avg_positions"] <= BCFG["max_positions"]


def test_rs_rating_bounds(random_prices):
    rs = bo.rs_rating(random_prices, 26).iloc[30:]
    assert rs.min().min() >= 1 and rs.max().max() <= 99
