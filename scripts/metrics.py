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
  * Any function returns None rather than NaN when input is insufficient, so
    the caller can distinguish "not enough data" from "computed as zero".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PPY = 52.0  # weekly periods per year


# ═══════════════════════════════════════════════════════════════════════════
# Momentum family
# ═══════════════════════════════════════════════════════════════════════════

def r2_trend(prices: np.ndarray) -> float | None:
    """
    R-squared of a log-linear trend fit. High R2 means the stock got where it
    got smoothly rather than via one gap — a proxy for trend persistence.
    Preserved from the original engine so historical scores stay comparable.
    Used as the scalar reference that panel.rolling_trend_r2's vectorised
    version is tested against.
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


# ═══════════════════════════════════════════════════════════════════════════
# Risk metrics
# ═══════════════════════════════════════════════════════════════════════════

def max_drawdown(prices: np.ndarray) -> float | None:
    """Worst peak-to-trough decline. Returned as a negative number."""
    p = prices[np.isfinite(prices)]
    if len(p) < 4:
        return None
    peak = np.maximum.accumulate(p)
    dd = p / peak - 1.0
    return float(np.min(dd))


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
