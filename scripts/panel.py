"""
panel.py
────────
Builds the full point-in-time factor panel: for every factor, a
(dates x tickers) DataFrame where cell [t, i] uses ONLY information available
at the close of week t.

Everything here is vectorised. The original engine recomputed an O(N) polyfit
inside a triple loop, which is why the client-side backtest could only afford
~60 weeks. Two identities make the whole thing collapse into rolling sums:

  1. For a log-linear trend fit against x = 0,1,...,w-1, the design matrix is
     constant. So R-squared is exactly the squared Pearson correlation between
     log-price and that fixed ramp — no polyfit needed, ever.
  2. Rolling beta is rolling_cov(y, x) / rolling_var(x), which pandas does in
     C. No per-window regression needed.

That takes a 500-name, 10-year panel from minutes to well under a second, so
a full-sample walk-forward evaluation becomes cheap enough to run every night.

LOOKAHEAD DISCIPLINE
Every value at row t is computed from prices at or before t. Forward returns
are produced separately in evaluation.py and are the ONLY forward-looking
object in the system. Any factor that needs prices after t is a bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import metrics as mx


def _wk(months: float, wpm: float) -> int:
    return int(round(months * wpm))


def rolling_trend_r2(log_prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    R-squared of a log-linear trend over a rolling window, via the fixed-ramp
    correlation identity. Bit-comparable to the original polyfit version on
    fully-populated windows; windows containing NaN yield NaN rather than
    silently fitting a shorter series.
    """
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    sxx = float(((x - x_mean) ** 2).sum())

    r = log_prices.rolling(window)
    y_sum = r.sum()
    y2_sum = (log_prices ** 2).rolling(window).sum()
    # sum(x_k * y_k) over the window, via a convolution with the ramp
    weighted = log_prices.rolling(window).apply(
        lambda v: float(np.dot(x, v)), raw=True, engine="numpy"
    ) if False else None
    # convolution is far faster than rolling.apply:
    arr = log_prices.to_numpy(dtype=float)
    T, N = arr.shape
    xy = np.full((T, N), np.nan)
    if T >= window:
        kernel = x[::-1]
        for j in range(N):
            col = arr[:, j]
            conv = np.convolve(col, kernel, mode="valid")  # length T-window+1
            xy[window - 1:, j] = conv
    xy = pd.DataFrame(xy, index=log_prices.index, columns=log_prices.columns)

    sxy = xy - x_mean * y_sum
    syy = y2_sum - (y_sum ** 2) / window
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = (sxy ** 2) / (sxx * syy)
    return r2.where(syy > 1e-12).clip(0.0, 1.0)


def rolling_beta_resid(stock_rets: pd.DataFrame, mkt_ret: pd.Series,
                       window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Point-in-time rolling market model. Returns (beta, idio_vol_ann, residuals).

    Residuals at time t use the beta/alpha estimated on the window ENDING at
    t — the coefficients an analyst would actually have had that day, not
    full-sample coefficients fitted with hindsight. Full-sample betas are a
    subtle and very common source of lookahead in factor research.
    """
    x = mkt_ret.reindex(stock_rets.index)
    x_mean = x.rolling(window).mean()
    x_var = x.rolling(window).var(ddof=1)
    y_mean = stock_rets.rolling(window).mean()
    cov = stock_rets.mul(x, axis=0).rolling(window).mean().sub(y_mean.mul(x_mean, axis=0))
    cov = cov * window / (window - 1)
    beta = cov.div(x_var, axis=0)
    alpha = y_mean.sub(beta.mul(x_mean, axis=0))
    resid = stock_rets.sub(beta.mul(x, axis=0)).sub(alpha)
    idio = resid.rolling(window).std(ddof=1) * np.sqrt(mx.PPY)
    return beta, idio, resid


def build_factor_panel(prices: pd.DataFrame, cfg: dict,
                       mkt_ret: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """
    Produce every factor as a (dates x tickers) panel.

    Returns a dict keyed by factor name. Factor sign convention: HIGHER IS
    ALWAYS BETTER. Factors whose natural direction is inverted (information
    discreteness, drawdown, volatility) are negated here so the composite can
    blend them without per-factor sign flags — a class of bug that is very
    easy to introduce and very hard to notice, because a sign-flipped factor
    still produces a plausible-looking ranking.
    """
    fc = cfg["factors"]
    wpm = fc["weeks_per_month"]
    skip = _wk(fc["skip_months"], wpm)
    hl = fc["vol_halflife_weeks"]

    logp = np.log(prices.where(prices > 0))
    rets = prices.pct_change()
    if mkt_ret is None:
        mkt_ret = rets.mean(axis=1)  # equal-weight universe proxy

    panels: dict[str, pd.DataFrame] = {}

    # ── Momentum: skip-adjusted total return over 12/9/6/3 months ──────────
    for label, months in [("r121", 12), ("r91", 9), ("r61", 6), ("r31", 3)]:
        lb = _wk(months, wpm)
        panels[label] = prices.shift(skip) / prices.shift(lb) - 1.0

    # ── Trend quality: R2 of log-linear fit ───────────────────────────────
    for label, months in [("r2_12", 12), ("r2_9", 9), ("r2_6", 6), ("r2_3", 3)]:
        # v1 used arr[pos-w : pos+1] -- an INCLUSIVE window of w+1 points.
        # Using w points here silently shifted every R2 value; the +1 makes
        # v2 reconcile exactly against the historical dashboard output.
        panels[label] = rolling_trend_r2(logp, _wk(months, wpm) + 1)

    # ── Volatility & risk ─────────────────────────────────────────────────
    vol_12 = rets.rolling(_wk(12, wpm)).std(ddof=1) * np.sqrt(mx.PPY)
    vol_3 = rets.rolling(_wk(3, wpm)).std(ddof=1) * np.sqrt(mx.PPY)
    ewma_var = rets.ewm(halflife=hl, min_periods=13).var(bias=False)
    panels["vol_12m"] = vol_12
    panels["ewma_vol"] = np.sqrt(ewma_var * mx.PPY)
    # Low-volatility anomaly: low-vol stocks earn higher risk-adjusted
    # returns. Negated so higher = better.
    panels["low_vol"] = -vol_12
    # Vol regime ratio > 1 means vol is expanding — the state in which
    # momentum crashes. Negated: calm trends score higher.
    panels["vol_regime"] = -(vol_3 / vol_12.replace(0, np.nan))

    # ── Risk-managed momentum (Barroso & Santa-Clara) ─────────────────────
    panels["mom_vol_scaled"] = panels["r121"] / vol_12.replace(0, np.nan)

    # ── 52-week high proximity (George & Hwang) ───────────────────────────
    hi52 = prices.rolling(52, min_periods=26).max()
    panels["pct_52w_high"] = prices / hi52.replace(0, np.nan)

    # ── Drawdown from running peak (negated: shallower is better) ─────────
    peak = prices.rolling(104, min_periods=26).max()
    panels["drawdown"] = prices / peak.replace(0, np.nan) - 1.0

    # ── Information discreteness (Frog-in-the-Pan) ────────────────────────
    w121 = _wk(12, wpm)
    pos_frac = (rets > 0).rolling(w121).mean()
    neg_frac = (rets < 0).rolling(w121).mean()
    cum = prices.shift(skip) / prices.shift(w121) - 1.0
    # LOW ID = continuous information = stronger momentum persistence,
    # so negate to keep "higher is better".
    panels["info_discreteness"] = -(np.sign(cum) * (neg_frac - pos_frac))

    # ── Consistency: fraction of positive weeks over 12m ──────────────────
    panels["consistency"] = pos_frac

    # ── Market model: beta, idio vol, residual momentum ───────────────────
    beta_w = min(52, max(26, len(prices) // 3))
    beta, idio, resid = rolling_beta_resid(rets, mkt_ret, beta_w)
    panels["beta"] = beta
    panels["idio_vol"] = idio
    panels["low_beta"] = -beta

    rm_w = _wk(12, wpm) - skip
    rm_mean = resid.shift(skip).rolling(rm_w).mean()
    rm_std = resid.shift(skip).rolling(rm_w).std(ddof=1)
    panels["resid_mom"] = (rm_mean / rm_std.replace(0, np.nan)) * np.sqrt(rm_w)

    # ── Short-term reversal (1 month, NOT skipped) ────────────────────────
    # Negated: recent losers tend to bounce. Kept as a separate sleeve
    # rather than folded into momentum, because it works on a different
    # horizon and blending them just cancels both out.
    panels["st_reversal"] = -(prices / prices.shift(skip) - 1.0)

    # ── Trailing risk-adjusted performance ────────────────────────────────
    panels["sharpe_12m"] = (rets.rolling(w121).mean() /
                            rets.rolling(w121).std(ddof=1).replace(0, np.nan)) * np.sqrt(mx.PPY)

    downside = rets.where(rets < 0, 0.0)
    dd_dev = np.sqrt((downside ** 2).rolling(w121).mean()) * np.sqrt(mx.PPY)
    panels["sortino_12m"] = (rets.rolling(w121).mean() * mx.PPY) / dd_dev.replace(0, np.nan)

    # ── Ulcer index (negated: less sustained pain is better) ──────────────
    dd_pct = (prices / prices.rolling(52, min_periods=26).max() - 1.0) * 100.0
    panels["ulcer"] = -np.sqrt((dd_pct ** 2).rolling(52, min_periods=26).mean())

    return panels


def composite_panel(panels: dict[str, pd.DataFrame], weights: dict,
                    sectors: pd.Series, cfg: dict) -> pd.DataFrame:
    """
    Blend factor panels into a single (dates x tickers) score panel, applying
    winsorisation, cross-sectional z-scoring and optional sector
    neutralisation date by date.
    """
    fc = cfg["factors"]
    keys = [k for k in weights if k in panels and weights[k] != 0]
    if not keys:
        raise ValueError("no active factors in weights")
    idx = panels[keys[0]].index
    cols = panels[keys[0]].columns
    out = pd.DataFrame(np.nan, index=idx, columns=cols)

    for dt in idx:
        frame = pd.DataFrame({k: panels[k].loc[dt] for k in keys})
        if frame.notna().sum().min() < 20:
            continue
        out.loc[dt] = mx.composite_score(
            frame, {k: weights[k] for k in keys}, sectors=sectors,
            winsor_sigma=fc["winsor_sigma"], sector_neutral=fc["sector_neutral"],
        )
    return out
