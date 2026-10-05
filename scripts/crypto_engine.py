"""
crypto_engine.py
────────────────
Pure, I/O-free building blocks for the crypto sleeve: weekly bar
construction, the factor panel, the point-in-time universe, the BTC regime
filter and the composite score. fetch_crypto.py and run_crypto.py do the I/O;
everything testable lives here.

Same rules as the equity engine (panel.py), restated because they matter more
in crypto, where data is messier and the universe turns over far faster:

  * LOOKAHEAD: every value at row t uses only prices/volumes at or before t.
    tests/test_crypto.py rebuilds every panel on truncated data and asserts
    the history is unchanged.
  * HIGHER IS BETTER for every factor, so the composite needs no sign flags.
  * The universe is decided week by week from what was knowable that week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import metrics as mx
import panel as pnl

FACTOR_KEYS = ["m1", "m2", "m3", "m6", "q2", "q3", "q6", "vadj"]
# Return / trend windows in WEEKS. Crypto runs on shorter horizons than the
# equity sleeve's 3-12 months, so the windows are set directly in weeks.
RET_WINDOWS = {"m1": 4, "m2": 8, "m3": 13, "m6": 26}
R2_WINDOWS = {"q2": 8, "q3": 13, "q6": 26}
VADJ_WINDOW = 13


# ═══════════════════════════════════════════════════════════════════════════
# Weekly bars
# ═══════════════════════════════════════════════════════════════════════════

def complete_weekly(daily: pd.DataFrame, rule: str, now: pd.Timestamp,
                    how: str = "last") -> pd.DataFrame:
    """
    Resample daily crypto bars to weekly, keeping ONLY closed weeks.

    Crypto never closes, so a run on Monday sees a partial Monday bar and
    Yahoo stamps it with today's date. Two cuts make the result independent
    of when the job happens to run:
      1. drop any daily bar dated today or later (still trading);
      2. drop the final weekly bucket if its label date is after the last
         remaining daily bar (the week has not finished).
    """
    now = pd.Timestamp(now).tz_localize(None).normalize()
    d = daily.copy()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    d = d[d.index < now]
    if d.empty:
        return d
    agg = d.resample(rule)
    w = agg.last() if how == "last" else agg.mean()
    if len(w) and w.index[-1] > d.index[-1]:
        w = w.iloc[:-1]
    return w


def detect_volume_units(btc_daily_volume: pd.Series, btc_daily_price: pd.Series) -> str:
    """
    Yahoo quotes crypto volume in USD, unlike equities (shares). Multiplying a
    USD volume by price inflates BTC's 'dollar volume' ~60,000x and turns the
    liquidity ranking into a price ranking. Decide from scale rather than
    trusting either convention: BTC trades ~1e5-1e6 coins/day but ~1e9-1e11
    USD/day, so the two are three-plus orders of magnitude apart.
    """
    v = btc_daily_volume.dropna().tail(90)
    p = btc_daily_price.dropna().tail(90)
    if v.empty or p.empty:
        return "usd"
    med_v = float(v.median())
    med_p = float(p.median())
    # If volume were in coins, volume*price would be a plausible USD figure
    # and volume itself far below it. If volume is already USD, it exceeds
    # any plausible coin count by a wide margin.
    return "coins" if med_v < 5e7 and med_v * med_p > 1e8 else "usd"


# ═══════════════════════════════════════════════════════════════════════════
# Factors
# ═══════════════════════════════════════════════════════════════════════════

def factor_panel(prices: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """
    The eight ranking factors plus display-only risk panels, each a
    (weeks x coins) frame. Every window ends at t - skip_weeks.

    m1/m2/m3/m6  simple return over 4/8/13/26 weeks
    q2/q3/q6     R² of a log-linear trend over 8/13/26 weeks (inclusive
                 window of w+1 points, matching the equity engine)
    vadj         13-week return divided by 13-week annualised volatility
    """
    skip = int(cfg["factors"].get("skip_weeks", 0))
    logp = np.log(prices.where(prices > 0))
    rets = prices.pct_change(fill_method=None)
    out: dict[str, pd.DataFrame] = {}

    for k, w in RET_WINDOWS.items():
        out[k] = prices.shift(skip) / prices.shift(skip + w) - 1.0
    for k, w in R2_WINDOWS.items():
        out[k] = pnl.rolling_trend_r2(logp, w + 1).shift(skip)

    vol = rets.rolling(VADJ_WINDOW).std(ddof=1).shift(skip) * np.sqrt(mx.PPY)
    out["vadj"] = out["m3"] / vol.replace(0, np.nan)

    # ── display-only (not ranked) ──────────────────────────────────────────
    out["vol_13w"] = rets.rolling(13).std(ddof=1) * np.sqrt(mx.PPY)
    hi = prices.rolling(52, min_periods=13).max()
    out["pct_52w_high"] = prices / hi.replace(0, np.nan)
    return out


def btc_beta(prices: pd.DataFrame, btc: str, window: int = 26) -> pd.DataFrame:
    """Rolling beta of each coin's weekly return to BTC's."""
    rets = prices.pct_change(fill_method=None)
    if btc not in rets.columns:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    beta, _, _ = pnl.rolling_beta_resid(rets, rets[btc], window)
    return beta


# ═══════════════════════════════════════════════════════════════════════════
# Universe
# ═══════════════════════════════════════════════════════════════════════════

def stablecoin_mask(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    True where a coin BEHAVES like a dollar-pegged asset over the trailing
    window: every price within ±dev of 1.00 and annualised vol below cap.
    Catches stablecoins nobody remembered to add to the exclude list. A coin
    that depegs (UST, May 2022) stops matching once it moves away from 1.0.
    """
    st = cfg["universe"]["stablecoin_test"]
    w = int(st["weeks"])
    dev = (prices - 1.0).abs().rolling(w, min_periods=max(4, w // 2)).max()
    vol = prices.pct_change(fill_method=None).rolling(
        w, min_periods=max(4, w // 2)).std(ddof=1) * np.sqrt(mx.PPY)
    return (dev <= st["max_abs_deviation_from_1"]) & (vol <= st["max_annual_vol"])


def universe_mask(prices: pd.DataFrame, dollar_volume: pd.DataFrame, cfg: dict,
                  factor_ok: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Point-in-time investable universe: a boolean (weeks x coins) frame.

    At week t a coin is in if, using only data up to t, it
      * has a price above min_price,
      * has at least min_history_weeks of observed prices (seasoning),
      * is not on the exclude list and does not behave like a stablecoin,
      * has trailing median daily dollar volume >= the floor,
      * has every ranking factor computable (if factor_ok is given),
    and then ranks inside the top `size` by that trailing dollar volume.
    """
    u = cfg["universe"]
    px = prices
    dv = dollar_volume.reindex(index=px.index, columns=px.columns)
    excluded = set(u.get("exclude", []))

    ok = px.notna() & (px > float(u.get("min_price", 0.0)))
    ok &= px.notna().cumsum() >= int(u["min_history_weeks"])
    ok &= ~stablecoin_mask(px, cfg).fillna(False)
    if excluded:
        ok.loc[:, [c for c in ok.columns if c in excluded]] = False
    w = int(u["dollar_volume_weeks"])
    med_dv = dv.rolling(w, min_periods=max(4, w // 2)).median()
    ok &= med_dv >= float(u["min_daily_dollar_volume"])
    if factor_ok is not None:
        ok &= factor_ok.reindex(index=px.index, columns=px.columns).fillna(False)

    ranked = med_dv.where(ok).rank(axis=1, ascending=False, method="first")
    return ok & (ranked <= int(u["size"]))


def factors_complete(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ok = None
    for k in FACTOR_KEYS:
        f = panels[k].notna() & np.isfinite(panels[k])
        ok = f if ok is None else (ok & f)
    return ok


def regime_series(prices: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    True (risk-on) where the regime asset's close is above its N-week SMA at
    that same close. NaN-safe: weeks before the SMA exists count as risk-on
    only if the filter is disabled; otherwise they are risk-off.
    """
    rc = cfg["regime"]
    if not rc.get("enabled", True) or rc["asset"] not in prices.columns:
        return pd.Series(True, index=prices.index)
    a = prices[rc["asset"]]
    sma = a.rolling(int(rc["sma_weeks"]), min_periods=int(rc["sma_weeks"])).mean()
    return (a > sma).fillna(False)


# ═══════════════════════════════════════════════════════════════════════════
# Composite
# ═══════════════════════════════════════════════════════════════════════════

def normalise_weights(weights: dict) -> dict:
    tot = sum(max(0.0, float(weights.get(k, 0))) for k in FACTOR_KEYS)
    return {k: (max(0.0, float(weights.get(k, 0))) / tot if tot > 0 else 0.0)
            for k in FACTOR_KEYS}


def composite_row(frame: pd.DataFrame, weights: dict, winsor_sigma: float) -> pd.Series:
    """
    One cross-section: winsorise -> z-score -> weighted sum -> re-z-score.
    Identical to metrics.composite_score with sector_neutral=False, and to
    the dashboard's JS scorer (tests/test_crypto.py checks the latter by
    running the dashboard's own function under node).
    """
    return mx.composite_score(frame, weights, sectors=None,
                              winsor_sigma=winsor_sigma, sector_neutral=False)


def composite_panel(panels: dict[str, pd.DataFrame], universe: pd.DataFrame,
                    weights: dict, cfg: dict, min_names: int = 10) -> pd.DataFrame:
    """Composite score per week over that week's universe only; NaN elsewhere."""
    ws = cfg["factors"]["winsor_sigma"]
    w = {k: v for k, v in normalise_weights(weights).items() if v > 0}
    out = pd.DataFrame(np.nan, index=universe.index, columns=universe.columns)
    for dt in universe.index:
        members = universe.columns[universe.loc[dt].to_numpy(dtype=bool)]
        if len(members) < min_names:
            continue
        frame = pd.DataFrame({k: panels[k].loc[dt, members] for k in w})
        out.loc[dt, members] = composite_row(frame, w, ws).to_numpy()
    return out
