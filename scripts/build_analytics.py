"""
build_analytics.py
──────────────────
Runs the full evaluation suite over the price panel and writes
data/analytics.json — the research tearsheet that sits behind the dashboard.

Run order in the pipeline:
    fetch_universe.py -> fetch_prices.py -> run_signals.py -> build_analytics.py

This script is the gate. If it reports an IC that is statistically
indistinguishable from zero, the correct response is not to reweight the
sliders until the backtest looks better — that is precisely the search process
that the Deflated Sharpe and PBO numbers below are designed to penalise.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import metrics as mx          # noqa: E402
import panel as pnl           # noqa: E402
import evaluation as ev       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config.yaml"

# Baseline weights = the legacy 8-factor equal blend, so the new analytics are
# directly comparable to what the dashboard has been showing all along.
LEGACY_WEIGHTS = {k: 1.0 for k in
                  ["r121", "r91", "r61", "r31", "r2_12", "r2_9", "r2_6", "r2_3"]}

# Candidate configurations evaluated for PBO. These are the "trials" — the
# count feeds the Deflated Sharpe deflation term, so it must be honest.
CANDIDATES = {
    "legacy_8factor": LEGACY_WEIGHTS,
    # Your dashboard's own default slider positions. Without this the tearsheet
    # grades a blend you never actually use, which makes the verdict answer a
    # question you did not ask.
    "dashboard_default": {"r121": 30, "r91": 25, "r61": 15, "r31": 10,
                          "r2_12": 8, "r2_9": 5, "r2_6": 5, "r2_3": 2},
    "pure_12_1": {"r121": 1.0},
    # The dashboard describes 9-1 as the best-Sharpe window, so it needs to be
    # testable on its own rather than assumed.
    "pure_9_1": {"r91": 1.0},
    "pure_3_1": {"r31": 1.0},
    "pure_6_1": {"r61": 1.0},
    "trend_quality_only": {"r2_12": 1.0, "r2_6": 1.0},
    "resid_mom": {"resid_mom": 1.0},
    "risk_managed_mom": {"mom_vol_scaled": 1.0},
    "52w_high": {"pct_52w_high": 1.0},
    "fip_momentum": {"r121": 1.0, "info_discreteness": 0.5},
    "low_vol": {"low_vol": 1.0},
    "quality_momentum": {"r121": 1.0, "sharpe_12m": 0.5, "low_vol": 0.3},
    "crash_protected": {"mom_vol_scaled": 1.0, "vol_regime": 0.5, "ulcer": 0.3},
    "multi_sleeve": {"resid_mom": 1.0, "pct_52w_high": 0.7, "info_discreteness": 0.4,
                     "low_vol": 0.3, "st_reversal": 0.2},
}


# PBO's resolution is limited by the number of trials: with only 12 candidates
# the out-of-sample rank statistic takes 12 discrete values, and the resulting
# PBO estimate is coarse enough that two unrelated markets can land on exactly
# the same number. Padding the trial set with randomised weight vectors over
# the same factor menu gives the statistic real resolution, and is also a
# fairer representation of the search space actually explored by anyone
# dragging weight sliders around until the backtest looks good.
RANDOM_TRIAL_FACTORS = ["r121", "r91", "r61", "r31", "r2_12", "r2_9", "r2_6",
                        "r2_3", "resid_mom", "mom_vol_scaled", "pct_52w_high",
                        "info_discreteness", "low_vol", "st_reversal",
                        "sharpe_12m", "vol_regime"]


def random_trials(n: int = 48, seed: int = 20260719) -> dict:
    """Random non-negative weight vectors over the factor menu, seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        k = int(rng.integers(2, 6))
        picks = rng.choice(RANDOM_TRIAL_FACTORS, size=k, replace=False)
        w = rng.random(k)
        out[f"rand_{i:03d}"] = {str(p): float(round(v, 3)) for p, v in zip(picks, w)}
    return out


def load_config() -> tuple[dict, str]:
    with open(CONFIG_PATH) as f:
        raw = f.read()
    return yaml.safe_load(raw), hashlib.sha256(raw.encode()).hexdigest()[:12]


def load_prices(market: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    path = DATA_DIR / f"{market}_prices.json"
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["prices"], index=pd.to_datetime(raw["dates"])).sort_index()
    sectors = pd.Series(raw.get("sectors", {}))
    return df, sectors, raw


def data_quality_report(prices: pd.DataFrame, raw: dict) -> dict:
    """
    Pre-flight checks. In a production shop this is a hard CI gate: bad data
    silently produces a plausible-looking ranking, which is worse than an
    outright crash because nobody investigates a number that looks fine.
    """
    rets = prices.pct_change()
    stale = (rets.abs() < 1e-9).rolling(4).sum().iloc[-1]
    n_stale = int((stale >= 4).sum())
    extreme = int((rets.abs() > 0.5).sum().sum())
    gaps = pd.Series(prices.index).diff().dt.days.value_counts().to_dict()

    return {
        "n_dates": int(len(prices)),
        "n_tickers": int(prices.shape[1]),
        "first_date": str(prices.index[0].date()),
        "last_date": str(prices.index[-1].date()),
        "missing_pct": round(float(prices.isna().mean().mean() * 100), 3),
        "tickers_stale_4w": n_stale,
        "weekly_moves_over_50pct": extreme,
        "irregular_date_gaps": {str(int(k)): int(v) for k, v in gaps.items() if k != 7.0},
        "universe_used_fallback": bool(raw.get("universeUsedFallback", False)),
        "fetched_at": raw.get("fetchedAt"),
        "survivorship_bias": (
            "PRESENT — the universe is today's index membership applied to all "
            "history. Delisted and demoted names are absent, which inflates "
            "every backtest statistic below. Magnitude is typically 1-4% p.a. "
            "for a large-cap index and materially more for a 300-name mid-cap "
            "index with real turnover. Fix requires point-in-time constituent "
            "history, which no free source provides."
        ),
    }


def evaluate_market(market: str, cfg: dict) -> dict:
    prices, sectors, raw = load_prices(market)
    ecfg, fcfg = cfg["evaluation"], cfg["factors"]

    min_p = cfg["universe"]["min_price"].get(market, 1.0)
    prices = prices.where(prices >= min_p)

    rets = prices.pct_change()
    mkt_ret = rets.mean(axis=1)
    benchmark_nav = (1.0 + mkt_ret.fillna(0.0)).cumprod()

    print(f"  building factor panel ({prices.shape[0]}w x {prices.shape[1]} names)...")
    panels = pnl.build_factor_panel(prices, cfg, mkt_ret=mkt_ret)

    out: dict = {
        "market": market,
        "data_quality": data_quality_report(prices, raw),
        "factors": {},
        "candidates": {},
    }

    # ── Per-factor IC across horizons ─────────────────────────────────────
    fwd = {h: ev.forward_returns(prices, h) for h in ecfg["forward_horizons_weeks"]}
    print("  computing information coefficients...")
    for name, pan in panels.items():
        rec = {"ic": {}}
        for h, f in fwd.items():
            r = ev.information_coefficient(pan, f, nw_lags=ecfg["newey_west_lags"])
            r.pop("ic_series", None)
            rec["ic"][f"{h}w"] = r
        rec["autocorr_4w"] = ev.signal_autocorrelation(pan, lag_w=4)
        out["factors"][name] = rec

    # ── Factor correlation matrix ─────────────────────────────────────────
    print("  factor correlation matrix...")
    core = {k: panels[k] for k in
            ["r121", "r91", "r61", "r31", "r2_12", "r2_6", "resid_mom",
             "mom_vol_scaled", "pct_52w_high", "low_vol", "info_discreteness",
             "st_reversal"] if k in panels}
    out["factor_correlation"] = ev.factor_correlation(core)

    # ── Candidate configurations ──────────────────────────────────────────
    print(f"  evaluating {len(CANDIDATES)} candidate configurations...")
    cost = (cfg["costs"]["spread_bps"][market] + cfg["costs"]["commission_bps"])
    trial_returns = {}

    named = set(CANDIDATES)
    all_trials = {**CANDIDATES, **random_trials(ecfg.get("n_random_trials", 48))}

    for cname, weights in all_trials.items():
        try:
            score = pnl.composite_panel(panels, weights, sectors, cfg)
        except Exception as e:  # a candidate referencing a missing factor
            if cname in named:
                out["candidates"][cname] = {"error": str(e)}
            continue

        if cname not in named:
            # Random trial: we need its return stream for PBO, nothing else.
            bt_r = ev.backtest_top_n(score, prices, top_n=cfg["portfolio"]["top_n"],
                                     rebalance_w=cfg["portfolio"]["rebalance_weeks"],
                                     cost_bps=cost)
            if "error" not in bt_r:
                trial_returns[cname] = pd.Series(bt_r["returns"], index=bt_r["dates"])
            continue

        h_primary = ecfg["forward_horizons_weeks"][1]
        ic = ev.information_coefficient(score, fwd[h_primary],
                                        nw_lags=ecfg["newey_west_lags"])
        ic_series = ic.pop("ic_series", [])
        buckets = ev.bucket_analysis(score, fwd[h_primary], ecfg["n_buckets"],
                                     nw_lags=ecfg["newey_west_lags"])
        bt = ev.backtest_top_n(score, prices, top_n=cfg["portfolio"]["top_n"],
                               rebalance_w=cfg["portfolio"]["rebalance_weeks"],
                               cost_bps=cost, benchmark=benchmark_nav)

        rec = {"weights": weights, "ic": ic, "buckets": buckets,
               "autocorr_4w": ev.signal_autocorrelation(score, lag_w=4)}
        if "error" not in bt:
            r = np.array(bt.pop("returns"))
            trial_returns[cname] = pd.Series(r, index=bt["dates"])
            rec["backtest"] = bt
            rec["deflated_sharpe"] = ev.deflated_sharpe(r, n_trials=ecfg["n_trials"])
            rec["ic_series"] = ic_series[-52:]
        else:
            rec["backtest"] = bt
        out["candidates"][cname] = rec

    # ── Probability of backtest overfitting across all candidates ─────────
    if len(trial_returns) >= 2:
        tr = pd.DataFrame(trial_returns)
        out["pbo"] = ev.pbo_cscv(tr, n_splits=8)
        best = tr.mean() / tr.std(ddof=1)
        out["selection"] = {
            "best_by_full_sample_sharpe": str(best.idxmax()),
            "sharpe_dispersion_across_trials": float(best.std(ddof=1)),
            "note": ("Full-sample selection is exactly the procedure PBO "
                     "measures. Use the walk-forward result, not this one."),
        }
    return out


def main() -> None:
    cfg, cfg_hash = load_config()
    out = {
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schemaVersion": cfg["schema_version"],
        "configHash": cfg_hash,
        "markets": {},
    }
    for market in ["sp500", "asx300"]:
        if not (DATA_DIR / f"{market}_prices.json").exists():
            print(f"  skipping {market}: no price file")
            continue
        print(f"Evaluating {market}...")
        out["markets"][market] = evaluate_market(market, cfg)

    path = DATA_DIR / "analytics.json"
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"), default=float)
    print(f"Saved {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()