"""
evaluation.py
─────────────
The part that separates a screener from a research process.

A screener tells you what ranks highest today. A research process tells you
whether the ranking has ever predicted anything, how confident you are allowed
to be in that, and how much of the edge survives contact with the market. This
module answers the second set of questions:

  * Information Coefficient — does the score correlate with FORWARD return?
    With autocorrelation-robust t-stats, because overlapping horizons make
    naive t-stats roughly sqrt(h) times too optimistic.
  * Decile spreads and monotonicity — is the effect real across the whole
    cross-section, or is one decile carrying it?
  * Turnover and breakeven cost — at what transaction cost does the edge die?
  * Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) — how much of the
    backtest Sharpe is explained by having tried many configurations?
  * Probability of Backtest Overfitting via CSCV — does in-sample rank
    predict out-of-sample rank, or is selection pure noise?

Nothing here is optional decoration. A momentum backtest run on 2 years of
post-hoc index membership will look spectacular and mean nothing; these are
the diagnostics that show it.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

from metrics import PPY

EULER = 0.5772156649015329


# ═══════════════════════════════════════════════════════════════════════════
# Robust inference
# ═══════════════════════════════════════════════════════════════════════════

def newey_west_tstat(x: np.ndarray, lags: int = 4) -> tuple[float, float]:
    """
    t-statistic for the mean of a serially-correlated series.

    Overlapping forward returns (a 26-week horizon sampled weekly) are heavily
    autocorrelated by construction. An OLS t-stat on that series is not
    slightly wrong, it is wrong by a factor of ~5. Newey-West with Bartlett
    kernel corrects it. Reporting an uncorrected t-stat on overlapping data is
    the single most common way an amateur backtest claims significance it
    does not have.

    Returns (mean, t_stat).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return float("nan"), float("nan")
    mu = float(np.mean(x))
    e = x - mu
    gamma0 = float(np.dot(e, e) / n)
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        cov = float(np.dot(e[l:], e[:-l]) / n)
        var += 2.0 * (1.0 - l / (lags + 1.0)) * cov
    var = max(var, 1e-18)
    se = math.sqrt(var / n)
    return mu, mu / se


def deflated_sharpe(returns: np.ndarray, n_trials: int, sr_benchmark: float = 0.0) -> dict:
    """
    Deflated Sharpe Ratio.

    The intuition: if you try 64 factor weightings, the best one has a high
    Sharpe *even when none of them has any edge*. DSR asks whether the
    observed Sharpe exceeds what the maximum of N independent noise trials
    would be expected to produce, correcting for the non-normality of returns.

    DSR is a probability. Below ~0.95 the strategy has not cleared the bar.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 20:
        return {"sharpe_ann": None, "dsr": None, "sr0_ann": None, "n_obs": n}
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-12:
        return {"sharpe_ann": None, "dsr": None, "sr0_ann": None, "n_obs": n}

    sr = float(np.mean(r)) / sd                      # per-period
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))      # non-excess

    # Expected maximum Sharpe across n_trials independent noise trials
    sigma_sr = math.sqrt(1.0 / max(n - 1, 1))
    N = max(int(n_trials), 2)
    z1 = stats.norm.ppf(1.0 - 1.0 / N)
    z2 = stats.norm.ppf(1.0 - 1.0 / (N * math.e))
    sr0 = sigma_sr * ((1.0 - EULER) * z1 + EULER * z2) + sr_benchmark

    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2
    if denom <= 0:
        return {"sharpe_ann": sr * math.sqrt(PPY), "dsr": None,
                "sr0_ann": sr0 * math.sqrt(PPY), "n_obs": n}
    dsr = float(stats.norm.cdf((sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)))
    return {
        "sharpe_ann": sr * math.sqrt(PPY),
        "sr0_ann": sr0 * math.sqrt(PPY),
        "dsr": dsr,
        "skew": g3,
        "kurtosis": g4,
        "n_obs": n,
        "n_trials": N,
    }


def pbo_cscv(trial_returns: pd.DataFrame, n_splits: int = 8) -> dict:
    """
    Probability of Backtest Overfitting, Combinatorially Symmetric CV
    (Bailey, Borwein, Lopez de Prado & Zhu 2015).

    Chop the return history into S blocks, take every way of splitting them
    into equal in-sample / out-of-sample halves, pick the best strategy on the
    IS half, and record where it lands in the OOS ranking. If selection is
    informative the winner stays near the top. If it is noise, the OOS rank is
    uniform and PBO tends to 0.5.

    PBO > 0.5 means your selection procedure is actively worse than random.

    trial_returns: DataFrame, one column per candidate configuration.
    """
    df = trial_returns.dropna(how="any")
    T, M = df.shape
    if M < 2 or T < n_splits * 4:
        return {"pbo": None, "reason": "insufficient data", "n_trials": M, "n_obs": T}

    if n_splits % 2:
        n_splits -= 1
    blocks = np.array_split(np.arange(T), n_splits)
    logits = []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        is_sr = df.iloc[is_idx].mean() / df.iloc[is_idx].std(ddof=1).replace(0, np.nan)
        oos_sr = df.iloc[oos_idx].mean() / df.iloc[oos_idx].std(ddof=1).replace(0, np.nan)
        if is_sr.isna().all() or oos_sr.isna().all():
            continue
        best = is_sr.idxmax()
        # relative rank of the IS winner within the OOS distribution
        rank = oos_sr.rank(pct=True).get(best, np.nan)
        if not np.isfinite(rank):
            continue
        w = min(max(float(rank), 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1.0 - w)))

    if not logits:
        return {"pbo": None, "reason": "no valid folds", "n_trials": M, "n_obs": T}
    logits = np.array(logits)
    return {
        "pbo": float(np.mean(logits <= 0.0)),
        "median_oos_logit": float(np.median(logits)),
        "n_folds": int(len(logits)),
        "n_trials": M,
        "n_obs": T,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Cross-sectional predictive power
# ═══════════════════════════════════════════════════════════════════════════

def forward_returns(prices: pd.DataFrame, horizon_w: int) -> pd.DataFrame:
    """Forward simple return over horizon_w weeks, aligned to the FORMATION date."""
    return prices.shift(-horizon_w) / prices - 1.0


def information_coefficient(scores: pd.DataFrame, fwd: pd.DataFrame,
                            nw_lags: int = 4, method: str = "spearman") -> dict:
    """
    Rank IC: cross-sectional Spearman correlation between score and forward
    return, computed each period, then summarised over time.

    Rules of thumb from the industry: |IC| of 0.02-0.04 is a real, tradeable
    equity signal. 0.05+ is excellent. Anything above 0.15 on a price-only
    signal means you have a lookahead bug, not an edge.

    ICIR = mean(IC) / std(IC) is the more important number — it measures
    CONSISTENCY, and it maps directly to strategy Sharpe via the fundamental
    law of active management: IR ≈ IC * sqrt(breadth).
    """
    common_rows = scores.index.intersection(fwd.index)
    common_cols = scores.columns.intersection(fwd.columns)
    s, f = scores.loc[common_rows, common_cols], fwd.loc[common_rows, common_cols]

    ics, dates, counts = [], [], []
    for dt in common_rows:
        a, b = s.loc[dt], f.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() < 20:
            continue
        if method == "spearman":
            ic, _ = stats.spearmanr(a[m], b[m])
        else:
            ic, _ = stats.pearsonr(a[m], b[m])
        if np.isfinite(ic):
            ics.append(float(ic))
            dates.append(dt)
            counts.append(int(m.sum()))

    if len(ics) < 8:
        return {"n_periods": len(ics), "ic_mean": None, "ic_ir": None, "t_stat": None}

    arr = np.array(ics)
    mean, t = newey_west_tstat(arr, lags=nw_lags)
    sd = float(np.std(arr, ddof=1))
    return {
        "n_periods": len(ics),
        "avg_breadth": float(np.mean(counts)),
        "ic_mean": mean,
        "ic_std": sd,
        "ic_ir": float(mean / sd) if sd > 1e-12 else None,
        "t_stat": float(t),
        "hit_rate": float(np.mean(arr > 0)),
        "ic_series": [{"date": str(d.date()), "ic": round(v, 4)} for d, v in zip(dates, arr)],
    }


def bucket_analysis(scores: pd.DataFrame, fwd: pd.DataFrame, n_buckets: int = 10,
                    nw_lags: int = 4) -> dict:
    """
    Sort into deciles each period, equal-weight within, track mean forward
    return per decile plus the long-short spread.

    Monotonicity is the honesty check. A genuine factor produces a roughly
    monotone staircase from decile 1 to decile 10. A spread that comes
    entirely from decile 10 outperforming, with deciles 1-9 flat, is usually
    a handful of microcaps or one lucky sector — not a factor.
    """
    common_rows = scores.index.intersection(fwd.index)
    common_cols = scores.columns.intersection(fwd.columns)
    s, f = scores.loc[common_rows, common_cols], fwd.loc[common_rows, common_cols]

    per_period = []
    for dt in common_rows:
        a, b = s.loc[dt], f.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() < n_buckets * 3:
            continue
        try:
            q = pd.qcut(a[m].rank(method="first"), n_buckets, labels=False)
        except ValueError:
            continue
        per_period.append(b[m].groupby(q).mean())

    if len(per_period) < 8:
        return {"n_periods": len(per_period), "buckets": None}

    panel = pd.DataFrame(per_period)
    means = panel.mean()
    spread = (panel[n_buckets - 1] - panel[0]).values
    sp_mean, sp_t = newey_west_tstat(spread, lags=nw_lags)
    mono, _ = stats.spearmanr(np.arange(len(means)), means.values)

    return {
        "n_periods": int(len(panel)),
        "buckets": [{"bucket": int(i) + 1, "mean_fwd_ret": round(float(v), 5)}
                    for i, v in means.items()],
        "spread_mean": float(sp_mean),
        "spread_t_stat": float(sp_t),
        "spread_hit_rate": float(np.mean(spread > 0)),
        "monotonicity_rho": float(mono) if np.isfinite(mono) else None,
    }


def factor_correlation(factor_panels: dict[str, pd.DataFrame]) -> dict:
    """
    Average cross-sectional rank correlation between factors.

    Eight momentum factors measured over overlapping windows are not eight
    bets — they are one bet counted eight times. If r121 and r91 correlate at
    0.9, splitting weight between them does nothing except create an illusion
    of diversification, and the composite's effective breadth is far lower
    than the slider count suggests.
    """
    keys = list(factor_panels.keys())
    out = {k: {} for k in keys}
    for a, b in itertools.combinations_with_replacement(keys, 2):
        pa, pb = factor_panels[a], factor_panels[b]
        rows = pa.index.intersection(pb.index)
        vals = []
        for dt in rows:
            x, y = pa.loc[dt], pb.loc[dt]
            m = x.notna() & y.notna()
            if m.sum() < 20:
                continue
            c, _ = stats.spearmanr(x[m], y[m])
            if np.isfinite(c):
                vals.append(c)
        v = round(float(np.mean(vals)), 3) if vals else None
        out[a][b] = v
        out[b][a] = v
    return out


def signal_autocorrelation(scores: pd.DataFrame, lag_w: int = 4) -> dict:
    """
    Rank autocorrelation of the score at a given lag. Drives expected
    turnover: turnover ≈ (1 - rank_autocorr), so a signal with 0.5 autocorr
    at your rebalance frequency will churn roughly half the book every
    rebalance. Combined with the cost model this tells you immediately
    whether the strategy can survive its own trading.
    """
    vals = []
    idx = list(scores.index)
    for i in range(lag_w, len(idx)):
        a, b = scores.loc[idx[i - lag_w]], scores.loc[idx[i]]
        m = a.notna() & b.notna()
        if m.sum() < 20:
            continue
        c, _ = stats.spearmanr(a[m], b[m])
        if np.isfinite(c):
            vals.append(c)
    if not vals:
        return {"lag_weeks": lag_w, "autocorr": None}
    ac = float(np.mean(vals))
    return {"lag_weeks": lag_w, "autocorr": ac, "implied_turnover_per_rebal": 1.0 - ac}


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio backtest
# ═══════════════════════════════════════════════════════════════════════════

def backtest_top_n(scores: pd.DataFrame, prices: pd.DataFrame, top_n: int = 5,
                   rebalance_w: int = 4, cost_bps: float = 30.0,
                   benchmark: pd.Series | None = None) -> dict:
    """
    Equal-weight top-N, rebalanced every rebalance_w weeks, with costs charged
    on realised turnover.

    Deliberately simple and deliberately pessimistic: positions are formed on
    the score at the close of week t and earn returns from t+1 onward, so
    there is no same-bar lookahead. Costs are charged on the sum of absolute
    weight changes at each rebalance.
    """
    rets = prices.pct_change()
    idx = list(scores.index)
    weights = pd.Series(dtype=float)
    port, turn, dates, held = [], [], [], []

    for i in range(len(idx) - 1):
        dt = idx[i]
        if i % rebalance_w == 0:
            row = scores.loc[dt].dropna()
            picks = row.nlargest(top_n).index if len(row) >= top_n else row.index
            new_w = pd.Series(1.0 / max(len(picks), 1), index=picks)
            allk = weights.index.union(new_w.index)
            t = float((new_w.reindex(allk).fillna(0) - weights.reindex(allk).fillna(0)).abs().sum())
            weights = new_w
        else:
            t = 0.0

        nxt = idx[i + 1]
        r = float((rets.loc[nxt].reindex(weights.index).fillna(0.0) * weights).sum())
        r -= t * cost_bps / 10000.0
        port.append(r)
        turn.append(t)
        dates.append(nxt)
        held.append(list(weights.index))

        # let winners run between rebalances (drifting weights, no free rebalance)
        gr = (1.0 + rets.loc[nxt].reindex(weights.index).fillna(0.0)) * weights
        if gr.sum() > 0:
            weights = gr / gr.sum()

    if len(port) < 12:
        return {"error": "insufficient history for backtest", "n_periods": len(port)}

    pr = np.array(port)
    curve = np.cumprod(1.0 + pr)
    years = len(pr) / PPY
    cagr = float(curve[-1] ** (1.0 / years) - 1.0) if years > 0 and curve[-1] > 0 else None
    peak = np.maximum.accumulate(curve)
    mdd = float(np.min(curve / peak - 1.0))
    sd = float(np.std(pr, ddof=1))
    sh = float(np.mean(pr) / sd * math.sqrt(PPY)) if sd > 1e-12 else None
    downside = pr[pr < 0]
    sortino_v = (float(np.mean(pr) * PPY / (np.std(downside, ddof=1) * math.sqrt(PPY)))
                 if len(downside) > 2 and np.std(downside, ddof=1) > 1e-12 else None)

    out = {
        "n_periods": len(pr),
        "years": round(years, 2),
        "cagr": cagr,
        "vol_ann": float(sd * math.sqrt(PPY)),
        "sharpe": sh,
        "sortino": sortino_v,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if cagr and mdd < -1e-9 else None,
        "hit_rate": float(np.mean(pr > 0)),
        "avg_turnover_per_rebal": float(np.mean([t for t in turn if t > 0])) if any(turn) else 0.0,
        "annual_turnover": float(np.sum(turn) / years) if years > 0 else None,
        "total_cost_drag_ann": float(np.sum(turn) * cost_bps / 10000.0 / years) if years > 0 else None,
        "equity_curve": [{"date": str(d.date()), "nav": round(float(v), 4)}
                         for d, v in zip(dates, curve)],
        "returns": pr.tolist(),
        "dates": [str(d.date()) for d in dates],
        "final_holdings": held[-1] if held else [],
    }

    # Breakeven cost: the round-trip bps at which gross alpha is exactly zero.
    gross_ann = float(np.mean(pr) * PPY) + (out["total_cost_drag_ann"] or 0.0)
    ann_turn = out["annual_turnover"]
    if ann_turn and ann_turn > 0:
        out["breakeven_cost_bps"] = float(gross_ann / ann_turn * 10000.0)

    if benchmark is not None:
        bm = benchmark.reindex(dates).pct_change().fillna(0.0).values
        if len(bm) == len(pr):
            active = pr - bm
            a_mean, a_t = newey_west_tstat(active, lags=4)
            te = float(np.std(active, ddof=1) * math.sqrt(PPY))
            out["benchmark"] = {
                "bm_cagr": float(np.prod(1 + bm) ** (1 / years) - 1) if years > 0 else None,
                "active_return_ann": float(a_mean * PPY),
                "tracking_error_ann": te,
                "information_ratio": float(a_mean * PPY / te) if te > 1e-9 else None,
                "active_t_stat": float(a_t),
            }
    return out
