"""
metrics.py
──────────
Pure, dependency-light metric library. Every function takes arrays/Series and
returns numbers — no I/O, no globals, no config reads. That makes the whole
file unit-testable, which is the difference between a signal you can defend in
a research review and one you hope is right.

Conventions used throughout:
  * All series are WEEKLY unless a name says otherwise.
  * PPY = 52 periods per year for annualisation.
  * Returns are simple (arithmetic) unless prefixed log_.
  * Any function returns None rather than NaN when input is insufficient, so
    the caller can distinguish "not enough data" from "computed as zero".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PPY = 52.0  # weekly periods per year


# ═══════════════════════════════════════════════════════════════════════════
# Return construction
# ═══════════════════════════════════════════════════════════════════════════

def simple_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change()


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))


# ═══════════════════════════════════════════════════════════════════════════
# Momentum family
# ═══════════════════════════════════════════════════════════════════════════

def skip_return(prices: np.ndarray, pos: int, lookback_w: int, skip_w: int) -> float | None:
    """
    Classic Jegadeesh-Titman momentum with a skip period: return from
    t-lookback to t-skip. Skipping the most recent month removes the
    short-term reversal effect that otherwise contaminates the signal.
    """
    if pos - lookback_w < 0 or pos - skip_w < 0:
        return None
    p0, p1 = prices[pos - lookback_w], prices[pos - skip_w]
    if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
        return None
    return float(p1 / p0 - 1.0)


def r2_trend(prices: np.ndarray) -> float | None:
    """
    R-squared of a log-linear trend fit. High R2 means the stock got where it
    got smoothly rather than via one gap — a proxy for trend persistence.
    Preserved from the original engine so historical scores stay comparable.
    """
    valid = prices[np.isfinite(prices)]
    if len(valid) < len(prices) * 0.6 or len(valid) < 6:
        return None
    y = np.log(valid)
    x = np.arange(len(y), dtype=float)
    if np.std(y) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return None
    return max(0.0, 1.0 - ss_res / ss_tot)


def information_discreteness(rets: np.ndarray) -> float | None:
    """
    Frog-in-the-Pan (Da, Gurun & Warachka 2014).

        ID = sign(cumulative return) * (%negative periods - %positive periods)

    Continuous information — a long string of small moves — is underreacted to
    by investors and its momentum persists. Discrete information (one huge jump)
    reverses. LOW ID is the desirable side: it flags smooth, gradual trends.
    Empirically this roughly doubles the Sharpe of a plain momentum book.
    """
    r = rets[np.isfinite(rets)]
    if len(r) < 12:
        return None
    cum = float(np.prod(1.0 + r) - 1.0)
    if cum == 0:
        return None
    pos = float(np.mean(r > 0))
    neg = float(np.mean(r < 0))
    return float(np.sign(cum) * (neg - pos))


def pct_from_52w_high(prices: np.ndarray, pos: int, window_w: int = 52) -> float | None:
    """
    George & Hwang (2004) 52-week-high momentum: price / 52-week high.
    Nearness to the high is a stronger and more persistent predictor than raw
    past return, and it is far less prone to momentum crashes because it does
    not load on stocks that have already gone parabolic.
    Returns a value in (0, 1]; 1.0 = at the high.
    """
    lo = max(0, pos - window_w + 1)
    window = prices[lo:pos + 1]
    window = window[np.isfinite(window)]
    if len(window) < 12:
        return None
    hi = float(np.max(window))
    p = prices[pos]
    if not np.isfinite(p) or hi <= 0:
        return None
    return float(p / hi)


def vol_scaled_momentum(mom: float | None, vol_ann: float | None) -> float | None:
    """
    Barroso & Santa-Clara (2015) risk-managed momentum: scale the momentum
    signal by its own realised volatility. Momentum's catastrophic crashes
    (1932, 2009, 2020) are concentrated in high-volatility states; dividing by
    vol removes most of the negative skew and roughly doubles the Sharpe.
    """
    if mom is None or vol_ann is None or vol_ann <= 1e-9:
        return None
    return float(mom / vol_ann)


# ═══════════════════════════════════════════════════════════════════════════
# Risk metrics
# ═══════════════════════════════════════════════════════════════════════════

def realised_vol(rets: np.ndarray, annualise: bool = True) -> float | None:
    r = rets[np.isfinite(rets)]
    if len(r) < 8:
        return None
    v = float(np.std(r, ddof=1))
    return v * np.sqrt(PPY) if annualise else v


def ewma_vol(rets: np.ndarray, halflife: float = 13.0) -> float | None:
    """
    Exponentially-weighted vol. Reacts to regime change far faster than a
    rolling window, which matters because the whole point of vol-scaling a
    momentum book is to de-risk *before* the crash, not after it.
    """
    r = rets[np.isfinite(rets)]
    if len(r) < 8:
        return None
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(len(r) - 1, -1, -1)
    w /= w.sum()
    mu = float(np.sum(w * r))
    var = float(np.sum(w * (r - mu) ** 2))
    return float(np.sqrt(max(var, 0.0) * PPY))


def downside_deviation(rets: np.ndarray, mar: float = 0.0) -> float | None:
    """Std of returns below a minimum acceptable return. Denominator of Sortino."""
    r = rets[np.isfinite(rets)]
    if len(r) < 8:
        return None
    d = np.minimum(r - mar, 0.0)
    return float(np.sqrt(np.mean(d ** 2)) * np.sqrt(PPY))


def max_drawdown(prices: np.ndarray) -> float | None:
    """Worst peak-to-trough decline. Returned as a negative number."""
    p = prices[np.isfinite(prices)]
    if len(p) < 4:
        return None
    peak = np.maximum.accumulate(p)
    dd = p / peak - 1.0
    return float(np.min(dd))


def ulcer_index(prices: np.ndarray) -> float | None:
    """
    RMS drawdown depth. Unlike max drawdown it penalises *sustained* pain
    rather than a single spike, so it distinguishes a fast -20%/recover from
    a two-year grind at -18%. Institutional risk teams care about the second.
    """
    p = prices[np.isfinite(prices)]
    if len(p) < 4:
        return None
    peak = np.maximum.accumulate(p)
    dd = (p / peak - 1.0) * 100.0
    return float(np.sqrt(np.mean(dd ** 2)))


def sharpe(rets: np.ndarray, rf_annual: float = 0.0) -> float | None:
    r = rets[np.isfinite(rets)]
    if len(r) < 12:
        return None
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-12:
        return None
    excess = float(np.mean(r)) - rf_annual / PPY
    return float(excess / sd * np.sqrt(PPY))


def sortino(rets: np.ndarray, rf_annual: float = 0.0) -> float | None:
    r = rets[np.isfinite(rets)]
    if len(r) < 12:
        return None
    dd = downside_deviation(r, mar=rf_annual / PPY)
    if dd is None or dd <= 1e-12:
        return None
    ann_excess = float(np.mean(r)) * PPY - rf_annual
    return float(ann_excess / dd)


def calmar(prices: np.ndarray, rets: np.ndarray) -> float | None:
    """Annualised return / |max drawdown|. The ratio allocators actually quote."""
    mdd = max_drawdown(prices)
    r = rets[np.isfinite(rets)]
    if mdd is None or mdd >= -1e-9 or len(r) < 12:
        return None
    ann = float(np.mean(r)) * PPY
    return float(ann / abs(mdd))


def tail_ratio(rets: np.ndarray) -> float | None:
    """95th percentile gain / |5th percentile loss|. >1 means positive tail skew."""
    r = rets[np.isfinite(rets)]
    if len(r) < 20:
        return None
    lo = float(np.percentile(r, 5))
    if abs(lo) < 1e-12:
        return None
    return float(abs(np.percentile(r, 95) / lo))


def hit_rate(rets: np.ndarray) -> float | None:
    r = rets[np.isfinite(rets)]
    if len(r) < 12:
        return None
    return float(np.mean(r > 0))


def skew_kurt(rets: np.ndarray) -> tuple[float | None, float | None]:
    r = rets[np.isfinite(rets)]
    if len(r) < 20:
        return None, None
    return float(stats.skew(r)), float(stats.kurtosis(r, fisher=True))


# ═══════════════════════════════════════════════════════════════════════════
# Market model: beta, alpha, idiosyncratic risk, residual momentum
# ═══════════════════════════════════════════════════════════════════════════

def market_model(stock_rets: np.ndarray, mkt_rets: np.ndarray) -> dict | None:
    """
    OLS of stock excess return on market return.

    Returns beta, annualised alpha, idiosyncratic (residual) vol, R2, and the
    residual series. Idio vol is the single most useful risk number here: two
    stocks with identical total vol can have wildly different diversification
    value, and equal-weighting the top 5 without looking at it is how you end
    up accidentally holding five expressions of the same trade.
    """
    m = np.isfinite(stock_rets) & np.isfinite(mkt_rets)
    y, x = stock_rets[m], mkt_rets[m]
    if len(y) < 20 or np.std(x) < 1e-12:
        return None
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (beta * x + alpha)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else None
    return {
        "beta": float(beta),
        "alpha_ann": float(alpha * PPY),
        "idio_vol": float(np.std(resid, ddof=1) * np.sqrt(PPY)),
        "r2_market": r2,
        "resid": resid,
        "n_obs": int(len(y)),
    }


def residual_momentum(stock_rets: np.ndarray, mkt_rets: np.ndarray,
                      lookback_w: int, skip_w: int) -> float | None:
    """
    Blitz, Huij & Martens (2011). Run the market model over the formation
    window, then compute momentum on the RESIDUALS, standardised by residual
    vol.

    Why this matters: conventional momentum is heavily contaminated by
    time-varying factor exposure — the winners are simply whatever had the
    highest beta to the sector that ran. Residual momentum strips that out and
    has historically delivered roughly double the risk-adjusted return with
    dramatically smaller crashes, because it does not implicitly short the
    market after a crash the way plain momentum does.
    """
    n = len(stock_rets)
    if n < lookback_w + 4:
        return None
    win_end = n - skip_w
    win_start = max(0, win_end - lookback_w)
    y = stock_rets[win_start:win_end]
    x = mkt_rets[win_start:win_end]
    mm = market_model(y, x)
    if mm is None:
        return None
    resid = mm["resid"]
    sd = float(np.std(resid, ddof=1))
    if sd <= 1e-12:
        return None
    return float(np.mean(resid) / sd * np.sqrt(len(resid)))


# ═══════════════════════════════════════════════════════════════════════════
# Cross-sectional transforms
# ═══════════════════════════════════════════════════════════════════════════

def winsorize(x: pd.Series, sigma: float = 3.0) -> pd.Series:
    """Clip at +/- sigma standard deviations. Applied BEFORE z-scoring."""
    mu, sd = x.mean(), x.std(ddof=1)
    if not np.isfinite(sd) or sd <= 1e-12:
        return x
    return x.clip(mu - sigma * sd, mu + sigma * sd)


def zscore(x: pd.Series, winsor_sigma: float | None = 3.0) -> pd.Series:
    """
    Cross-sectional z-score. Preferred over percentile rank for blending
    because ranks throw away magnitude — a stock 4 sigma ahead of the field
    and one 0.1 sigma ahead both become "rank 1", and the composite then
    cannot tell a strong month from a flat one.
    """
    v = winsorize(x, winsor_sigma) if winsor_sigma else x
    mu, sd = v.mean(), v.std(ddof=1)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(0.0, index=x.index)
    return (v - mu) / sd


def percentile_rank(x: pd.Series) -> pd.Series:
    """
    (count_below + 0.5) / n — bit-for-bit identical to the existing dashboard
    engine, retained so v2 scores can be reconciled against v1.
    """
    return x.rank(method="average", pct=True) - 0.5 / max(len(x), 1)


def sector_neutralize(x: pd.Series, sectors: pd.Series, min_members: int = 5) -> pd.Series:
    """
    Demean within GICS sector, then re-standardise. Sectors with fewer than
    min_members are pooled into an 'other' bucket — demeaning a 2-stock sector
    just sets both to zero and destroys real information.
    """
    grp = sectors.reindex(x.index).fillna("—")
    counts = grp.value_counts()
    small = counts[counts < min_members].index
    grp = grp.where(~grp.isin(small), "__other__")
    out = x.groupby(grp).transform(lambda g: g - g.mean())
    sd = out.std(ddof=1)
    return out / sd if np.isfinite(sd) and sd > 1e-12 else out


def composite_score(factor_frame: pd.DataFrame, weights: dict,
                    sectors: pd.Series | None = None,
                    winsor_sigma: float = 3.0,
                    sector_neutral: bool = True) -> pd.Series:
    """
    Blend factors into one score: winsorise -> z-score -> (sector-neutralise)
    -> weighted sum -> re-standardise. Weights are renormalised over whichever
    factors are actually present, so a missing column degrades gracefully
    instead of silently down-weighting the whole composite.
    """
    present = [k for k in weights if k in factor_frame.columns and weights[k] != 0]
    if not present:
        return pd.Series(0.0, index=factor_frame.index)
    total = sum(abs(weights[k]) for k in present)
    acc = pd.Series(0.0, index=factor_frame.index)
    for k in present:
        z = zscore(factor_frame[k].astype(float), winsor_sigma)
        if sector_neutral and sectors is not None:
            z = sector_neutralize(z, sectors)
        acc = acc.add(z * (weights[k] / total), fill_value=0.0)
    return zscore(acc, winsor_sigma=None)
