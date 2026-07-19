"""
portfolio.py
────────────
Turns a score into positions.

The gap between "top 5 by score" and an actual portfolio is where most of the
real-world return goes. Equal-weighting the five highest-scoring names in a
momentum book in 2026 typically produces five semiconductor stocks with a
combined beta near 1.8 and a pairwise correlation of 0.8 — a single levered
sector bet wearing the costume of a diversified portfolio. It will look
brilliant right up until it does not.

Constraints applied, in order:
  1. Liquidity     — no position larger than a set share of median daily
                     dollar volume. This is a hard reality constraint, not a
                     preference: a backtest that assumes fills it cannot get
                     is fiction.
  2. Sector cap    — bounds concentration in one GICS sector.
  3. Position cap  — bounds single-name risk.
  4. Vol targeting — scales gross exposure so ex-ante portfolio volatility
                     hits a target, using a shrunk covariance matrix. This is
                     what converts a momentum strategy's fat left tail into
                     something an allocator will hold through a drawdown.

Cost model is square-root market impact plus half-spread plus commission,
which is the standard practitioner form and, importantly, makes cost scale
with size — so the model knows that a $50k position and a $50m position in
the same microcap are not the same trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import PPY


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf shrinkage toward a constant-correlation target.

    With 60 weekly observations and 5-20 assets the sample covariance matrix
    is badly conditioned, and any optimiser fed a raw sample covariance will
    happily load up on whichever pair happens to look most negatively
    correlated by chance. Shrinkage pulls the estimate toward a structured
    target and is the difference between a risk model and a random number
    generator.

    Returns (covariance, shrinkage_intensity).
    """
    X = returns.dropna(axis=1, how="all").dropna()
    n, p = X.shape
    if n < 10 or p < 2:
        return np.diag(returns.var(ddof=1).fillna(0.0).values), 1.0

    S = np.cov(X.values, rowvar=False, ddof=1)
    var = np.diag(S)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = S / np.outer(sd, sd)
    off = R[~np.eye(p, dtype=bool)]
    rbar = float(np.nanmean(off)) if off.size else 0.0
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, var)

    Xc = X.values - X.values.mean(axis=0)
    # asymptotic variance of the sample covariance entries
    phi = 0.0
    for t in range(n):
        d = np.outer(Xc[t], Xc[t]) - S
        phi += float(np.sum(d ** 2))
    phi /= n
    gamma = float(np.sum((F - S) ** 2))
    delta = float(np.clip(phi / (n * gamma), 0.0, 1.0)) if gamma > 1e-18 else 1.0
    return delta * F + (1 - delta) * S, delta


def transaction_cost_bps(participation: float, spread_bps: float,
                         commission_bps: float, impact_coef_bps: float) -> float:
    """
    One-way cost estimate.

        cost = half_spread + commission + impact_coef * sqrt(participation)

    Square-root impact is the well-established empirical form (Almgren et al.);
    it means doubling your size costs 1.41x, not 2x, per share — but it also
    means the cost is unbounded as you approach the full day's volume, which
    is exactly the discipline a small-account backtest needs to inherit before
    it is scaled up.
    """
    p = max(float(participation), 0.0)
    return spread_bps / 2.0 + commission_bps + impact_coef_bps * np.sqrt(p)


def construct(scores: pd.Series, sectors: pd.Series, cfg: dict,
              returns: pd.DataFrame | None = None,
              dollar_volume: pd.Series | None = None,
              capital: float = 100_000.0) -> pd.DataFrame:
    """
    Build a target book from a single cross-section of scores.

    Returns a DataFrame indexed by ticker with the weight and the binding
    constraint for each name, so every position can be explained. "Why is my
    weight 12% and not 20%?" should always have an answer; an allocator will
    ask, and "the code did it" is not one.
    """
    pc = cfg["portfolio"]
    top_n = pc["top_n"]

    s = scores.dropna().sort_values(ascending=False)
    if s.empty:
        return pd.DataFrame(columns=["weight", "binding_constraint"])

    # ── 0. Constrained selection ──────────────────────────────────────────
    # The sector cap has to bind at SELECTION, not at weighting. Scaling the
    # weights of five semiconductor stocks down still leaves a portfolio that
    # is 100% semiconductors — it just makes it a smaller one. The cap only
    # means anything if it changes which names get held, so walk down the
    # ranking and skip names whose sector is already full.
    max_per_sector = max(1, int(np.floor(pc["max_sector_weight"] * top_n + 1e-9)))
    sec_all = sectors.reindex(s.index).fillna("—")
    picks, counts, skipped = [], {}, []
    for t in s.index:
        if len(picks) >= top_n:
            break
        sname = sec_all.get(t, "—")
        if counts.get(sname, 0) >= max_per_sector:
            skipped.append((t, sname))
            continue
        picks.append(t)
        counts[sname] = counts.get(sname, 0) + 1

    if not picks:
        return pd.DataFrame(columns=["weight", "binding_constraint"])

    w = pd.Series(1.0 / len(picks), index=picks)
    binding = pd.Series("equal_weight", index=picks)

    # ── 1. Liquidity cap ──────────────────────────────────────────────────
    if dollar_volume is not None:
        adv = dollar_volume.reindex(picks)
        cap = (adv * pc["max_adv_participation"]) / capital
        hit = cap.notna() & (cap < w)
        w[hit] = cap[hit]
        binding[hit] = "liquidity"

    # ── 2. Position cap ───────────────────────────────────────────────────
    hit = w > pc["max_weight"]
    w[hit] = pc["max_weight"]
    binding[hit] = "position_cap"

    # ── 3. Sector cap residual check ─────────────────────────────────────
    # Selection already enforces the count cap. Position/liquidity caps can
    # still push the *weight* distribution around, so trim any sector that
    # ends up over the limit and re-mark the binding constraint.
    sec = sectors.reindex(picks).fillna("—")
    for _ in range(10):
        tot = w.sum()
        if tot <= 0:
            break
        sw = (w / tot).groupby(sec).sum()
        over = sw[sw > pc["max_sector_weight"] + 1e-9]
        if over.empty:
            break
        for sector_name, weight in over.items():
            members = sec[sec == sector_name].index
            w[members] *= pc["max_sector_weight"] / weight
            binding[members] = "sector_cap"

    # ── 4. Ex-ante volatility targeting ──────────────────────────────────
    # Cap gross exposure at 1.0 but do NOT scale it back UP to 1.0. If the
    # liquidity or position caps have cut total exposure to 30%, the other
    # 70% is cash — renormalising to fully invested would quietly reverse
    # every constraint applied above, which is exactly the bug this comment
    # exists to prevent recurring.
    gross = float(w.sum())
    if gross > 1.0:
        w = w / gross
    ex_ante_vol = None
    if returns is not None and pc.get("vol_target_annual"):
        sub = returns.reindex(columns=picks).dropna(axis=1, how="all")
        if sub.shape[1] >= 2:
            cov, shrink = ledoit_wolf_shrinkage(sub)
            wv = w.reindex(sub.columns).fillna(0.0).values
            ex_ante_vol = float(np.sqrt(max(wv @ cov @ wv, 0.0)) * np.sqrt(PPY))
            if ex_ante_vol > 1e-9:
                scale = min(pc["vol_target_annual"] / ex_ante_vol, 1.0)
                w = w * scale
                if scale < 0.999:
                    binding[:] = binding.where(binding != "equal_weight", "vol_target")

    out = pd.DataFrame({
        "score": s.reindex(picks),
        "sector": sec,
        "weight": w,
        "binding_constraint": binding,
    })
    out.attrs["displaced_by_sector_cap"] = skipped[:10]
    out.attrs["max_names_per_sector"] = max_per_sector
    out.attrs["cash_weight"] = float(max(0.0, 1.0 - w.sum()))
    out.attrs["ex_ante_vol_annual"] = ex_ante_vol
    out.attrs["gross_exposure"] = float(w.sum())
    return out.sort_values("weight", ascending=False)


def risk_report(book: pd.DataFrame, returns: pd.DataFrame,
                market_returns: pd.Series) -> dict:
    """
    Post-construction risk summary. What an allocator asks for before they
    ask about returns: how concentrated is it, what is it actually exposed
    to, and how much of the risk is diversifiable.
    """
    w = book["weight"]
    sub = returns.reindex(columns=w.index).dropna(how="all", axis=1)
    if sub.shape[1] < 2:
        return {"error": "insufficient return history"}

    wv = w.reindex(sub.columns).fillna(0.0)
    port = (sub * wv).sum(axis=1)
    cov, shrink = ledoit_wolf_shrinkage(sub)
    vec = wv.values
    port_var = float(vec @ cov @ vec)

    # marginal and component contribution to risk
    mctr = (cov @ vec) / np.sqrt(port_var) if port_var > 1e-18 else np.zeros_like(vec)
    cctr = vec * mctr
    total = cctr.sum()

    m = port.notna() & market_returns.reindex(port.index).notna()
    beta = None
    if m.sum() > 20:
        x = market_returns.reindex(port.index)[m].values
        y = port[m].values
        if np.std(x) > 1e-12:
            beta = float(np.polyfit(x, y, 1)[0])

    corr = sub.corr()
    iu = np.triu_indices_from(corr.values, k=1)
    weights_norm = w / w.sum() if w.sum() > 0 else w

    return {
        "ex_ante_vol_annual": float(np.sqrt(max(port_var, 0)) * np.sqrt(PPY)),
        "shrinkage_intensity": float(shrink),
        "portfolio_beta": beta,
        "effective_n": float(1.0 / (weights_norm ** 2).sum()) if w.sum() > 0 else None,
        "herfindahl": float((weights_norm ** 2).sum()) if w.sum() > 0 else None,
        "avg_pairwise_correlation": float(np.nanmean(corr.values[iu])),
        "max_pairwise_correlation": float(np.nanmax(corr.values[iu])),
        "sector_weights": weights_norm.groupby(book["sector"]).sum().round(4).to_dict(),
        "risk_contribution": {t: round(float(c / total), 4)
                              for t, c in zip(sub.columns, cctr)} if total > 1e-18 else {},
        "concentration_warning": (
            "Effective N below 3 means this is a concentrated bet, not a "
            "portfolio — position sizing should reflect that."
            if w.sum() > 0 and 1.0 / (weights_norm ** 2).sum() < 3 else None
        ),
    }
