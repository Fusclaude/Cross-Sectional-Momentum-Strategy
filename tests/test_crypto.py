"""
test_crypto.py
──────────────
Correctness tests for the crypto sleeve. Run with: python3 -m pytest tests/ -q

As with test_engine.py, the one that matters most is the point-in-time
truncation test: rebuild every panel (factors, universe, regime, composite)
on data cut off at week T and assert the history is identical to the
full-sample build. The universe is the new risk here -- a liquidity rank
computed with any future data would quietly select coins for having
survived, which is exactly the bias the sleeve is designed to avoid.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import crypto_engine as ce  # noqa: E402
import metrics as mx        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg():
    c = yaml.safe_load((ROOT / "config_crypto.yaml").read_text())
    c["universe"]["size"] = 15
    c["universe"]["min_daily_dollar_volume"] = 1e6
    return c


@pytest.fixture(scope="module")
def market():
    """40 coins x 160 weeks with staggered listings, one stablecoin, one crash."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2022-01-02", periods=160, freq="W-SUN")
    px, dv = {}, {}
    btc_r = rng.normal(0.004, 0.06, len(idx))
    for j in range(40):
        t = "BTC-USD" if j == 0 else f"C{j:02d}-USD"
        r = (1.0 if j == 0 else rng.uniform(0.7, 1.5)) * btc_r + rng.normal(0, 0.05, len(idx))
        p = 5 * np.exp(np.cumsum(np.log1p(np.clip(r, -0.9, 2))))
        start = 0 if j < 10 else int(rng.integers(0, 100))
        p[:start] = np.nan
        px[t] = p
        dv[t] = np.exp(rng.normal(np.log(2e7), 1.0, len(idx)))
    # LUNA-style collapse: -98% in one week, then keeps trading noisily.
    px["C05-USD"][90:] = px["C05-USD"][89] * 0.02 * np.exp(
        np.cumsum(rng.normal(0, 0.08, len(idx) - 90)))
    px["PEG-USD"] = 1 + rng.normal(0, 0.001, len(idx))  # unlisted stablecoin
    dv["PEG-USD"] = np.full(len(idx), 1e10)               # most liquid of all
    return pd.DataFrame(px, index=idx), pd.DataFrame(dv, index=idx)


def _build(px, dv, cfg):
    p = ce.factor_panel(px, cfg)
    u = ce.universe_mask(px, dv, cfg, factor_ok=ce.factors_complete(p))
    r = ce.regime_series(px, cfg)
    s = ce.composite_panel(p, u, cfg["default_weights"], cfg, min_names=5)
    return p, u, r, s


# ═══════════════════════════════════════════════════════════════════════════
# Lookahead
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("cut", [70, 110, 140])
def test_point_in_time_truncation(market, cfg, cut):
    px, dv = market
    full = _build(px, dv, cfg)
    part = _build(px.iloc[:cut], dv.iloc[:cut], cfg)
    rows = px.index[:cut]
    for k in ce.FACTOR_KEYS:
        pd.testing.assert_frame_equal(full[0][k].loc[rows], part[0][k], check_freq=False,
                                      obj=f"factor {k}")
    pd.testing.assert_frame_equal(full[1].loc[rows], part[1], check_freq=False, obj="universe")
    pd.testing.assert_series_equal(full[2].loc[rows], part[2], check_freq=False, obj="regime")
    pd.testing.assert_frame_equal(full[3].loc[rows], part[3], check_freq=False,
                                  obj="composite", atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# Universe
# ═══════════════════════════════════════════════════════════════════════════

def test_universe_respects_size_and_floor(market, cfg):
    px, dv = market
    _, u, _, _ = _build(px, dv, cfg)
    assert int(u.sum(axis=1).max()) <= cfg["universe"]["size"]
    med = dv.rolling(cfg["universe"]["dollar_volume_weeks"], min_periods=6).median()
    assert (med.where(u) >= cfg["universe"]["min_daily_dollar_volume"]).where(u).fillna(True).all().all()


def test_unlisted_stablecoin_caught_by_behaviour(market, cfg):
    px, dv = market
    assert "PEG-USD" not in cfg["universe"]["exclude"]
    _, u, _, _ = _build(px, dv, cfg)
    assert not u["PEG-USD"].any(), "pegged coin entered the universe"


def test_named_exclusion(market, cfg):
    px, dv = market
    c = yaml.safe_load(yaml.safe_dump(cfg))
    c["universe"]["exclude"] = ["BTC-USD"]
    _, u, _, _ = _build(px, dv, c)
    assert not u["BTC-USD"].any()


def test_seasoning(market, cfg):
    px, dv = market
    _, u, _, _ = _build(px, dv, cfg)
    seen = px.notna().cumsum()
    assert (seen.where(u) >= cfg["universe"]["min_history_weeks"]).where(u).fillna(True).all().all()


def test_collapsed_coin_is_not_deleted(market, cfg):
    """A coin that crashes 98% must stay rankable while it is liquid -- dropping
    it is survivorship bias in the other direction."""
    px, dv = market
    p, u, _, _ = _build(px, dv, cfg)
    after = px.index[95:]
    assert ce.factors_complete(p).loc[after, "C05-USD"].all()


# ═══════════════════════════════════════════════════════════════════════════
# Bars, units, regime
# ═══════════════════════════════════════════════════════════════════════════

def test_complete_weekly_drops_partial_day_and_week():
    days = pd.date_range("2026-09-21", "2026-10-07", freq="D")   # Mon .. Wed
    daily = pd.DataFrame({"X": np.arange(len(days), dtype=float)}, index=days)
    w = ce.complete_weekly(daily, "W-SUN", pd.Timestamp("2026-10-07 01:00"))
    # Weeks ending 27 Sep and 4 Oct are closed; the week ending 11 Oct is not.
    assert list(w.index.strftime("%Y-%m-%d")) == ["2026-09-27", "2026-10-04"]
    assert w["X"].iloc[-1] == daily.loc["2026-10-04", "X"]
    # Running on Monday 5 Oct must give the same answer as Wednesday.
    w2 = ce.complete_weekly(daily, "W-SUN", pd.Timestamp("2026-10-05 00:30"))
    pd.testing.assert_frame_equal(w, w2)


def test_volume_units_detection():
    p = pd.Series(np.full(90, 60_000.0))
    assert ce.detect_volume_units(pd.Series(np.full(90, 3e10)), p) == "usd"
    assert ce.detect_volume_units(pd.Series(np.full(90, 5e5)), p) == "coins"


def test_regime_is_same_bar(cfg):
    idx = pd.date_range("2024-01-07", periods=60, freq="W-SUN")
    a = np.r_[np.linspace(100, 200, 40), np.linspace(200, 80, 20)]
    px = pd.DataFrame({"BTC-USD": a}, index=idx)
    r = ce.regime_series(px, cfg)
    sma = px["BTC-USD"].rolling(cfg["regime"]["sma_weeks"]).mean()
    expect = (px["BTC-USD"] > sma).fillna(False)
    pd.testing.assert_series_equal(r, expect, check_names=False)
    assert not r.iloc[: cfg["regime"]["sma_weeks"] - 1].any()


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard parity: the page ranks with its own JS. It must agree with Python.
# ═══════════════════════════════════════════════════════════════════════════

def _js_scorer() -> str:
    html = (ROOT / "docs" / "crypto" / "index.html").read_text()
    m = re.search(r"// BEGIN SCORER\n(.*?)// END SCORER", html, re.S)
    assert m, "scorer markers missing from docs/crypto/index.html"
    return m.group(1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_dashboard_scorer_matches_python(market, cfg):
    px, dv = market
    p, u, _, _ = _build(px, dv, cfg)
    dt = px.index[-1]
    members = list(u.columns[u.loc[dt].to_numpy(dtype=bool)])
    frame = pd.DataFrame({k: p[k].loc[dt, members] for k in ce.FACTOR_KEYS})
    weights = {"m1": 10, "m2": 0, "m3": 35, "m6": 20, "q2": 5, "q3": 0, "q6": 10, "vadj": 20}
    w = {k: v for k, v in ce.normalise_weights(weights).items() if v > 0}
    py = ce.composite_row(frame, w, cfg["factors"]["winsor_sigma"])

    rows = [[float(frame.loc[t, k]) for k in ce.FACTOR_KEYS] for t in members]
    script = _js_scorer() + f"""
const out = compositeScores({json.dumps(rows)}, {json.dumps(ce.FACTOR_KEYS)},
                            {json.dumps(weights)}, {cfg['factors']['winsor_sigma']});
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    js = np.array(json.loads(res.stdout))
    np.testing.assert_allclose(js, py.to_numpy(), atol=1e-9)
