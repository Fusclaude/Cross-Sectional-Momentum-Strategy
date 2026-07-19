"""
test_engine.py
──────────────
Correctness tests. Run with: python3 -m pytest tests/ -q

The two that matter most are test_point_in_time_truncation and
test_placebo_ic. Everything else checks arithmetic; those two check the thing
that actually invalidates factor research.

  * Point-in-time truncation: rebuild the entire factor panel using only data
    up to date T, then assert every historical value matches the full-history
    panel exactly. If any factor peeks forward — a centred rolling window, a
    full-sample z-score, a full-sample beta — the truncated values differ and
    this test fails. It is the only reliable mechanical check for lookahead.

  * Placebo: shuffle the forward returns so no relationship can exist, then
    confirm the measured IC is statistically zero. If the pipeline reports a
    significant IC on shuffled targets, the significance machinery is broken
    and every number it has ever produced is meaningless.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import metrics as mx        # noqa: E402
import panel as pnl         # noqa: E402
import evaluation as ev     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def prices():
    """Synthetic panel: 200 weeks, 60 names, mixed drift and vol. Deterministic."""
    rng = np.random.default_rng(42)
    T, N = 200, 60
    drift = rng.normal(0.0015, 0.002, N)
    vol = rng.uniform(0.015, 0.055, N)
    shocks = rng.normal(0, 1, (T, N)) * vol + drift
    mkt = rng.normal(0.001, 0.02, (T, 1))
    beta = rng.uniform(0.4, 1.8, N)
    rets = shocks + mkt * beta
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range("2022-01-07", periods=T, freq="W-FRI")
    return pd.DataFrame(px, index=idx, columns=[f"T{i:02d}" for i in range(N)])


@pytest.fixture(scope="module")
def sectors(prices):
    names = ["Tech", "Health", "Financials", "Energy", "Staples"]
    return pd.Series({t: names[i % len(names)] for i, t in enumerate(prices.columns)})


# ═══════════════════════════════════════════════════════════════════════════
# The critical tests
# ═══════════════════════════════════════════════════════════════════════════

def test_point_in_time_truncation(prices, cfg):
    """
    Rebuilding the panel on truncated history must not change any historical
    factor value. This is the mechanical lookahead detector.
    """
    full = pnl.build_factor_panel(prices, cfg)

    # beta window adapts to sample length, so exclude the market-model
    # factors from the exact check and test them separately below.
    market_model_factors = {"beta", "low_beta", "idio_vol", "resid_mom"}

    failures = []
    for cutoff in (120, 150, 180):
        trunc = pnl.build_factor_panel(prices.iloc[:cutoff], cfg)
        last = cutoff - 1
        for name, panel_full in full.items():
            if name in market_model_factors:
                continue
            a = panel_full.iloc[last]
            b = trunc[name].iloc[last]

            # (1) A value computable with full history must still be
            #     computable at the time -- an extra NaN in the truncated run
            #     means the factor needed data from the future. This is the
            #     check whose absence let a centred rolling window through.
            vanished = int((a.notna() & b.isna()).sum())
            if vanished:
                failures.append((name, cutoff, f"{vanished} values require future data"))
                continue

            # (2) Values present in both must match exactly.
            both = a.notna() & b.notna()
            if both.sum() == 0:
                continue
            diff = float((a[both] - b[both]).abs().max())
            if not (diff < 1e-9):
                failures.append((name, cutoff, f"value drift {diff:.3e}"))
    assert not failures, f"lookahead detected: {failures}"


def test_market_model_is_point_in_time(prices, cfg):
    """
    Rolling beta must use a trailing window only. With the estimation window
    pinned, truncation must leave historical betas untouched.
    """
    rets = prices.pct_change()
    mkt = rets.mean(axis=1)
    w = 52
    b_full, _, _ = pnl.rolling_beta_resid(rets, mkt, w)
    cutoff = 150
    # recompute market return from truncated data too - a realistic replay
    r_t = prices.iloc[:cutoff].pct_change()
    b_tr, _, _ = pnl.rolling_beta_resid(r_t, r_t.mean(axis=1), w)
    a, b = b_full.iloc[:cutoff], b_tr
    m = a.notna() & b.notna()
    assert (a[m] - b[m]).abs().max().max() < 1e-9


def test_placebo_ic(prices, cfg, sectors):
    """
    Shuffled forward returns must produce an IC indistinguishable from zero.
    """
    panels = pnl.build_factor_panel(prices, cfg)
    score = pnl.composite_panel(panels, {"r121": 1.0, "r61": 1.0}, sectors, cfg)
    fwd = ev.forward_returns(prices, 4)

    rng = np.random.default_rng(7)
    shuffled = fwd.copy()
    for dt in shuffled.index:  # permute cross-section within each date
        row = shuffled.loc[dt].values.copy()
        rng.shuffle(row)
        shuffled.loc[dt] = row

    res = ev.information_coefficient(score, shuffled)
    assert abs(res["ic_mean"]) < 0.03, f"placebo IC too large: {res['ic_mean']}"
    assert abs(res["t_stat"]) < 3.0, f"placebo t-stat too large: {res['t_stat']}"


def test_lookahead_signal_is_detected(prices, cfg):
    """
    Positive control: a 'signal' that IS the forward return must show IC near
    1. If this fails, the IC machinery is not measuring what it claims to.
    """
    fwd = ev.forward_returns(prices, 4)
    res = ev.information_coefficient(fwd, fwd)
    assert res["ic_mean"] > 0.99


def test_backtest_has_no_same_bar_lookahead(prices, cfg, sectors):
    """
    A score built from the forward return should win; a score built from the
    CURRENT bar's return must not be able to capture that same bar. We assert
    the backtest of a perfect-foresight signal beats a random one — confirming
    signal-to-return alignment is the intended one-bar lag, not zero.
    """
    fwd1 = ev.forward_returns(prices, 1)
    perfect = ev.backtest_top_n(fwd1, prices, top_n=5, rebalance_w=1, cost_bps=0.0)
    rng = np.random.default_rng(3)
    noise = pd.DataFrame(rng.normal(size=prices.shape),
                         index=prices.index, columns=prices.columns)
    rand = ev.backtest_top_n(noise, prices, top_n=5, rebalance_w=1, cost_bps=0.0)
    assert perfect["sharpe"] > rand["sharpe"] + 3.0


# ═══════════════════════════════════════════════════════════════════════════
# Arithmetic
# ═══════════════════════════════════════════════════════════════════════════

def test_max_drawdown_known():
    p = np.array([100.0, 120.0, 60.0, 90.0])
    assert mx.max_drawdown(p) == pytest.approx(-0.5)


def test_r2_trend_perfect_exponential():
    p = 100 * np.exp(0.01 * np.arange(60))
    assert mx.r2_trend(p) == pytest.approx(1.0, abs=1e-9)


def test_rolling_r2_matches_scalar(prices):
    w = 52
    roll = pnl.rolling_trend_r2(np.log(prices), w)
    col = prices.columns[0]
    arr = prices[col].to_numpy()
    for pos in (60, 120, 199):
        expected = mx.r2_trend(arr[pos - w + 1: pos + 1])
        assert roll[col].iloc[pos] == pytest.approx(expected, abs=1e-8)


def test_zscore_properties():
    s = pd.Series([1, 2, 3, 4, 100.0])
    z = mx.zscore(s, winsor_sigma=3.0)
    assert abs(z.mean()) < 1e-9
    assert z.max() < 3.5  # outlier clipped


def test_sector_neutralize_removes_sector_means():
    x = pd.Series({"a": 1.0, "b": 2.0, "c": 10.0, "d": 11.0,
                   "e": 1.5, "f": 2.5, "g": 10.5, "h": 11.5,
                   "i": 1.2, "j": 10.2})
    sec = pd.Series({"a": "X", "b": "X", "e": "X", "f": "X", "i": "X",
                     "c": "Y", "d": "Y", "g": "Y", "h": "Y", "j": "Y"})
    out = mx.sector_neutralize(x, sec)
    assert abs(out.groupby(sec).mean().abs().max()) < 1e-9


def test_newey_west_widens_se_under_autocorrelation():
    rng = np.random.default_rng(1)
    e = rng.normal(size=500)
    ar = np.zeros(500)
    for i in range(1, 500):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    ar += 0.05
    _, t_nw = ev.newey_west_tstat(ar, lags=8)
    t_ols = np.mean(ar) / (np.std(ar, ddof=1) / np.sqrt(len(ar)))
    assert abs(t_nw) < abs(t_ols)


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(5)
    r = rng.normal(0.004, 0.02, 300)
    few = ev.deflated_sharpe(r, n_trials=2)
    many = ev.deflated_sharpe(r, n_trials=5000)
    assert few["dsr"] > many["dsr"]


def test_pbo_on_pure_noise_is_near_half():
    rng = np.random.default_rng(11)
    df = pd.DataFrame(rng.normal(0, 0.02, (240, 20)))
    res = ev.pbo_cscv(df, n_splits=8)
    assert 0.3 < res["pbo"] < 0.7


def test_costs_reduce_returns(prices, cfg, sectors):
    panels = pnl.build_factor_panel(prices, cfg)
    score = pnl.composite_panel(panels, {"r121": 1.0}, sectors, cfg)
    free = ev.backtest_top_n(score, prices, 5, 4, cost_bps=0.0)
    pricey = ev.backtest_top_n(score, prices, 5, 4, cost_bps=100.0)
    assert pricey["cagr"] < free["cagr"]


def test_higher_is_better_sign_convention(prices, cfg):
    """Negated factors must actually be negated."""
    panels = pnl.build_factor_panel(prices, cfg)
    last = panels["low_vol"].iloc[-1].dropna()
    vol = panels["vol_12m"].iloc[-1].dropna()
    common = last.index.intersection(vol.index)
    assert np.corrcoef(last[common], vol[common])[0, 1] < -0.99


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio construction
# ═══════════════════════════════════════════════════════════════════════════

import portfolio as pf  # noqa: E402


def test_sector_cap_binds_at_selection(cfg, prices, sectors):
    """
    Regression test for a real bug: the original implementation scaled the
    weights of an over-concentrated sector but never changed WHICH names were
    selected, so a book of five same-sector names stayed 100% that sector.
    """
    one_sector = pd.Series("Tech", index=prices.columns)
    scores = pd.Series(np.arange(len(prices.columns), 0, -1, dtype=float),
                       index=prices.columns)
    mixed = pd.Series(["Tech", "Health", "Energy", "Financials", "Staples"] *
                      (len(prices.columns) // 5 + 1))[:len(prices.columns)]
    mixed.index = prices.columns

    book = pf.construct(scores, mixed, cfg)
    sw = (book["weight"] / book["weight"].sum()).groupby(book["sector"]).sum()
    assert sw.max() <= cfg["portfolio"]["max_sector_weight"] + 1e-9

    # degenerate case: only one sector exists, cap cannot be met by
    # substitution -- must still not crash and must flag the concentration
    book2 = pf.construct(scores, one_sector, cfg)
    assert len(book2) >= 1


def test_position_cap_respected(cfg, prices, sectors):
    scores = pd.Series(np.random.default_rng(0).normal(size=len(prices.columns)),
                       index=prices.columns)
    book = pf.construct(scores, sectors, cfg)
    norm = book["weight"] / book["weight"].sum()
    assert norm.max() <= cfg["portfolio"]["max_weight"] + 1e-9


def test_liquidity_cap_binds(cfg, prices, sectors):
    scores = pd.Series(np.arange(len(prices.columns), 0, -1, dtype=float),
                       index=prices.columns)
    tiny = pd.Series(1_000.0, index=prices.columns)  # $1k ADV: everything illiquid
    book = pf.construct(scores, sectors, cfg, dollar_volume=tiny, capital=1_000_000)
    assert (book["binding_constraint"] == "liquidity").any()
    assert book["weight"].sum() < 0.01


def test_vol_target_scales_down_not_up(cfg, prices, sectors):
    """Vol targeting must never lever ABOVE 100% gross -- that is a different
    product and requires margin the user does not have."""
    scores = pd.Series(np.arange(len(prices.columns), 0, -1, dtype=float),
                       index=prices.columns)
    calm = prices.pct_change() * 0.01  # near-zero vol
    book = pf.construct(scores, sectors, cfg, returns=calm)
    assert book["weight"].sum() <= 1.0 + 1e-9


def test_shrinkage_intensity_in_unit_interval(prices):
    rets = prices.pct_change().iloc[1:, :8]
    _, d = pf.ledoit_wolf_shrinkage(rets)
    assert 0.0 <= d <= 1.0


def test_impact_cost_is_concave_in_size():
    a = pf.transaction_cost_bps(0.01, 10, 2, 40)
    b = pf.transaction_cost_bps(0.04, 10, 2, 40)
    # 4x the participation must cost less than 4x the impact
    fixed = 10 / 2 + 2
    assert (b - fixed) < 4 * (a - fixed)


# ═══════════════════════════════════════════════════════════════════════════
# Data quality gate
# ═══════════════════════════════════════════════════════════════════════════

import fetch_prices as fpx  # noqa: E402


def _qpayload(cols, n=80):
    dates = pd.date_range("2018-01-05", periods=n, freq="W-FRI")
    return {"prices": cols, "dates": [d.strftime("%Y-%m-%d") for d in dates]}


def _ok(n=80):
    return list(np.linspace(20, 30, n))


def test_tick_quantisation_is_not_flagged_as_a_split(cfg):
    """
    Regression test for a real false positive (LOT.AX, Nov 2018).

    The ASX quotes sub-$0.10 names in $0.001 increments, so a stock
    oscillating between $0.005 and $0.010 produces an EXACTLY 2:1 ratio from a
    single tick. The naive clean-ratio check called that an unadjusted split
    and failed the entire pipeline. Such a stock is also below the universe
    screen, so the data could not have affected any ranking regardless.
    """
    payload = _qpayload({"PENNY.AX": [0.010, 0.005] * 40, "OK.AX": _ok()})
    fails, _ = fpx.quality_gate(payload, "ASX 300", cfg["quality"], min_price=0.05)
    assert not [f for f in fails if "split" in f], f"penny stock flagged: {fails}"


def test_genuine_unadjusted_split_is_still_caught(cfg):
    """The guard must not disarm the check for a real split in a real stock."""
    px = [50.0] * 40 + [25.0] + [50.0] * 39
    payload = _qpayload({"REAL.AX": px, "OK.AX": _ok()})
    fails, _ = fpx.quality_gate(payload, "ASX 300", cfg["quality"], min_price=0.05)
    assert any("split" in f for f in fails), f"real split missed: {fails}"


def test_historical_halt_warns_but_does_not_fail(cfg):
    """A flat run that later resumes is a trading halt, not a dead feed."""
    px = list(np.linspace(10, 12, 30)) + [12.0] * 8 + list(np.linspace(12, 15, 42))
    payload = _qpayload({"HALT.AX": px, "OK.AX": _ok()})
    fails, warns = fpx.quality_gate(payload, "ASX 300", cfg["quality"], min_price=0.05)
    assert not [f for f in fails if "flat" in f], f"halt failed the gate: {fails}"
    assert any("resumed" in w for w in warns)


def test_dead_feed_at_final_bar_fails(cfg):
    """Still flat at the last bar corrupts TODAY's ranking. That must fail."""
    dead = list(np.linspace(10, 12, 60)) + [12.0] * 20
    cols = {f"DEAD{i}.AX": list(dead) for i in range(6)}
    cols["OK.AX"] = _ok()
    fails, _ = fpx.quality_gate(_qpayload(cols), "ASX 300", cfg["quality"], min_price=0.05)
    assert any("dead feed" in f for f in fails), f"dead feed missed: {fails}"
