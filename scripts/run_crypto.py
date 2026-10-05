"""
run_crypto.py
─────────────
Computes the crypto sleeve and writes data/crypto_signals.json, which the
crypto dashboard (docs/crypto/index.html) reads.

Same division of labour as the equity sleeve: Python computes FACTS (factor
values, point-in-time universe, regime, history, research statistics) and the
dashboard does the RANKING, so the weight sliders stay live.

What it writes:
  coins[]       latest-week universe with factors, risk, charts
  book          reference Top-N under config default_weights
  regime        BTC trend filter state now and over the chart window
  history       52 weeks of factor vectors for the dashboard's rank grid
  research      IC / quintile / backtest / deflated Sharpe / PBO, computed on
                the full point-in-time history under the default weights
  manifest      config hash, git rev, input digest
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crypto_engine as ce  # noqa: E402
import evaluation as ev     # noqa: E402
import metrics as mx        # noqa: E402
from picks_history_writer import upsert_picks  # noqa: E402
from run_signals import file_digest, git_rev, monthly_history  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IN = DATA_DIR / "crypto_prices.json"
OUT = DATA_DIR / "crypto_signals.json"
HIST = DATA_DIR / "crypto_picks_history.jsonl"


def load_config() -> tuple[dict, str]:
    raw = (ROOT / "config_crypto.yaml").read_text()
    return yaml.safe_load(raw), hashlib.sha256(raw.encode()).hexdigest()[:12]


def fnum(v, nd: int = 6):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = json.loads(IN.read_text())
    idx = pd.to_datetime(raw["dates"])
    px = pd.DataFrame(raw["prices"], index=idx, dtype=float).sort_index()
    dv = pd.DataFrame(raw["dollarVolume"], index=idx, dtype=float).reindex(
        columns=px.columns).sort_index()
    return px, dv, raw


# ═══════════════════════════════════════════════════════════════════════════
# Research
# ═══════════════════════════════════════════════════════════════════════════

def equal_weight_universe_returns(prices: pd.DataFrame, uni: pd.DataFrame) -> pd.Series:
    """Weekly return of an equal-weight book of LAST week's universe."""
    r = prices.pct_change(fill_method=None)
    held = uni.shift(1, fill_value=False).astype(bool)
    return r.where(held).mean(axis=1)


def run_trials(panels, uni, regime, prices, cfg) -> dict:
    """
    Every configuration this script evaluates: the default blend and each
    factor on its own, each with and without the regime filter. All share
    one date axis starting at the first week the universe is wide enough, so
    weeks spent in cash count as 0% rather than silently shortening a trial.
    """
    pc, ec = cfg["portfolio"], cfg["evaluation"]
    min_names = int(ec["min_names"])
    wide = uni.sum(axis=1) >= min_names
    if not wide.any():
        return {}
    start = wide.idxmax()

    configs = {"blend": cfg["default_weights"]}
    configs.update({k: {k: 1.0} for k in ce.FACTOR_KEYS})

    btc = prices[cfg["regime"]["asset"]] if cfg["regime"]["asset"] in prices else None
    trials, details = {}, {}
    for name, w in configs.items():
        sc = ce.composite_panel(panels, uni, w, cfg, min_names=min_names).loc[start:]
        for filt in (False, True):
            s = sc.copy()
            if filt:
                s.loc[~regime.reindex(s.index).fillna(False)] = np.nan
            label = f"{name}{'+regime' if filt else ''}"
            bt = ev.backtest_top_n(s, prices.loc[start:], top_n=int(pc["top_n"]),
                                   rebalance_w=int(pc["rebalance_weeks"]),
                                   cost_bps=float(pc["one_way_cost_bps"]),
                                   benchmark=btc.loc[start:] if btc is not None else None)
            if "returns" not in bt:
                continue
            trials[label] = pd.Series(bt["returns"], index=pd.to_datetime(bt["dates"]))
            if name == "blend":
                details[label] = bt
    axis = prices.loc[start:].index[1:]
    frame = pd.DataFrame({k: v.reindex(axis) for k, v in trials.items()}).fillna(0.0)
    return {"frame": frame, "details": details, "start": start}


def summarise_backtest(bt: dict) -> dict:
    keep = ["n_periods", "years", "cagr", "vol_ann", "sharpe", "sortino", "max_drawdown",
            "calmar", "hit_rate", "annual_turnover", "total_cost_drag_ann",
            "breakeven_cost_bps", "annual_returns", "drawdown_detail", "benchmark"]
    return {k: bt.get(k) for k in keep}


def curve(rets: pd.Series) -> list[float]:
    return [round(float(v), 4) for v in (1.0 + rets.fillna(0.0)).cumprod()]


def research(panels, uni, regime, prices, cfg) -> dict:
    ec = cfg["evaluation"]
    out: dict = {}
    score = ce.composite_panel(panels, uni, cfg["default_weights"], cfg,
                               min_names=int(ec["min_names"]))

    # ── Predictive power of the default blend ─────────────────────────────
    ic = {}
    for h in ec["forward_horizons_weeks"]:
        fwd = ev.forward_returns(prices, int(h)).where(uni)
        # Overlapping h-week returns are autocorrelated to ~h lags.
        res = ev.information_coefficient(score, fwd, nw_lags=max(int(ec["newey_west_lags"]), int(h)))
        if h != 4:
            res.pop("ic_series", None)   # ship one IC series (4w) to keep the payload small
        ic[f"{h}w"] = res
    out["ic"] = ic

    fwd4 = ev.forward_returns(prices, 4).where(uni)
    out["factor_ic_4w"] = {}
    for k in ce.FACTOR_KEYS:
        r = ev.information_coefficient(panels[k].where(uni), fwd4, nw_lags=4)
        out["factor_ic_4w"][k] = {x: r.get(x) for x in ("ic_mean", "t_stat", "ic_ir", "hit_rate", "n_periods")}
    out["quintiles_4w"] = ev.bucket_analysis(score, fwd4, n_buckets=int(ec["n_buckets"]), nw_lags=4)
    out["autocorr"] = ev.signal_autocorrelation(score, lag_w=int(cfg["portfolio"]["rebalance_weeks"]))
    out["factor_corr"] = ev.factor_correlation({k: panels[k].where(uni) for k in ce.FACTOR_KEYS})

    # ── Backtests and overfitting statistics ──────────────────────────────
    t = run_trials(panels, uni, regime, prices, cfg)
    if not t:
        out["backtest"] = {"error": "universe never reached min_names"}
        return out
    frame, details = t["frame"], t["details"]
    n_trials = max(int(ec["n_trials"]), frame.shape[1])

    bt = {}
    for label, d in details.items():
        s = summarise_backtest(d)
        s["deflated_sharpe"] = ev.deflated_sharpe(frame[label].to_numpy(), n_trials)
        bt[label] = s
    out["backtest"] = bt
    out["pbo"] = ev.pbo_cscv(frame, n_splits=8)
    out["trials"] = {c: {"sharpe": fnum(frame[c].mean() / frame[c].std(ddof=1) * np.sqrt(mx.PPY), 3)
                         if frame[c].std(ddof=1) > 0 else None,
                         "cagr": fnum((1 + frame[c]).prod() ** (mx.PPY / len(frame)) - 1, 4)}
                     for c in frame.columns}
    out["n_trials_declared"] = n_trials

    # ── Curves for the dashboard chart, one shared axis ───────────────────
    axis = frame.index
    btc = prices[cfg["regime"]["asset"]].pct_change(fill_method=None).reindex(axis)
    ew = equal_weight_universe_returns(prices, uni).reindex(axis)
    out["curves"] = {
        "dates": [d.strftime("%Y-%m-%d") for d in axis],
        "blend": curve(frame.get("blend", pd.Series(0.0, index=axis))),
        "blend_regime": curve(frame.get("blend+regime", pd.Series(0.0, index=axis))),
        "btc": curve(btc),
        "ew_universe": curve(ew),
        "regime": [bool(x) for x in regime.reindex(axis).shift(1, fill_value=False)],
    }
    out["start"] = t["start"].strftime("%Y-%m-%d")
    out["universe_size_series"] = [int(x) for x in uni.loc[t["start"]:].sum(axis=1)]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg, cfg_hash = load_config()
    prices, dv, raw = load_prices()
    names, cats = raw.get("names", {}), raw.get("categories", {})

    panels = ce.factor_panel(prices, cfg)
    uni = ce.universe_mask(prices, dv, cfg, factor_ok=ce.factors_complete(panels))
    regime = ce.regime_series(prices, cfg)
    beta = ce.btc_beta(prices, cfg["regime"]["asset"])
    med_dv = dv.rolling(int(cfg["universe"]["dollar_volume_weeks"]), min_periods=4).median()

    asof = prices.index[-1]
    members = list(uni.columns[uni.loc[asof].to_numpy(dtype=bool)])
    print(f"Crypto as of {asof.date()}: {len(members)} coins in universe, "
          f"regime {'RISK-ON' if regime.loc[asof] else 'RISK-OFF'}")

    # Reference ranking under the default weights (the dashboard recomputes
    # this live from the same factor values).
    w = ce.normalise_weights(cfg["default_weights"])
    frame_now = pd.DataFrame({k: panels[k].loc[asof, members] for k in ce.FACTOR_KEYS})
    score_now = ce.composite_row(frame_now, {k: v for k, v in w.items() if v > 0},
                                 cfg["factors"]["winsor_sigma"]).sort_values(ascending=False)

    chart_w = int(cfg["dashboard"]["chart_weeks"])
    coins = []
    for t in members:
        p = prices[t]
        coins.append({
            "ticker": t,
            "symbol": t.split("-")[0].rstrip("0123456789") or t,
            "name": names.get(t, t),
            "category": cats.get(t, "—"),
            "price": fnum(p.loc[asof], 8),
            "factors": {k: fnum(panels[k].loc[asof, t]) for k in ce.FACTOR_KEYS},
            "risk": {
                "vol_13w": fnum(panels["vol_13w"].loc[asof, t], 4),
                "pct_52w_high": fnum(panels["pct_52w_high"].loc[asof, t], 4),
                "beta_btc": fnum(beta.loc[asof, t], 3) if t in beta.columns else None,
                "max_drawdown_1y": fnum(mx.max_drawdown(p.tail(52).to_numpy()), 4),
                "median_daily_dollar_volume": fnum(med_dv.loc[asof, t], 0),
            },
            "defaultScore": fnum(score_now.get(t), 4),
            "monthly": monthly_history(p),
            "chart": [{"date": d.strftime("%Y-%m-%d"), "price": fnum(v, 8)}
                      for d, v in p.tail(chart_w).items()],
        })
    coins.sort(key=lambda c: -(c["defaultScore"] if c["defaultScore"] is not None else -9))

    top_n = int(cfg["portfolio"]["top_n"])
    book = [{"ticker": t, "name": names.get(t, t), "weight": round(1.0 / top_n, 4),
             "score": fnum(score_now[t], 4)} for t in score_now.index[:top_n]]

    # ── History for the rank grid / chart overlay ─────────────────────────
    hw = int(cfg["dashboard"]["history_weeks"])
    hdates = prices.index[-hw:]
    hist_coins = {}
    for t in prices.columns:
        inu = uni.loc[hdates, t].to_numpy(dtype=bool)
        if not inu.any():
            continue
        rows = []
        for i, d in enumerate(hdates):
            rows.append([fnum(panels[k].loc[d, t]) for k in ce.FACTOR_KEYS] if inu[i] else None)
        hist_coins[t] = rows

    rc = cfg["regime"]
    a = prices[rc["asset"]]
    sma = a.rolling(int(rc["sma_weeks"]), min_periods=int(rc["sma_weeks"])).mean()

    print("Running research evaluation...")
    res = research(panels, uni, regime, prices, cfg)

    out = {
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schemaVersion": cfg["schema_version"],
        "asOf": asof.strftime("%Y-%m-%d"),
        "manifest": {
            "configHash": cfg_hash,
            "gitRev": git_rev(),
            "inputs": {"crypto_prices.json": file_digest(IN)},
            "volumeUnits": raw.get("volumeUnits"),
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
        },
        "config": {
            "factorKeys": ce.FACTOR_KEYS,
            "defaultWeights": cfg["default_weights"],
            "winsorSigma": cfg["factors"]["winsor_sigma"],
            "skipWeeks": cfg["factors"]["skip_weeks"],
            "topN": top_n,
            "rebalanceWeeks": cfg["portfolio"]["rebalance_weeks"],
            "oneWayCostBps": cfg["portfolio"]["one_way_cost_bps"],
            "universeSize": cfg["universe"]["size"],
            "minDailyDollarVolume": cfg["universe"]["min_daily_dollar_volume"],
            "regimeAsset": rc["asset"], "regimeSmaWeeks": rc["sma_weeks"],
            "regimeEnabled": rc.get("enabled", True),
        },
        "universeSize": len(members),
        "candidateCount": len(cfg["universe"]["candidates"]),
        "missingTickers": raw.get("missingTickers", []),
        "qualityProblems": raw.get("qualityProblems", []),
        "qualityWarnings": raw.get("qualityWarnings", []),
        "regime": {
            "riskOn": bool(regime.loc[asof]),
            "price": fnum(a.loc[asof], 2),
            "sma": fnum(sma.loc[asof], 2),
            "distance": fnum(a.loc[asof] / sma.loc[asof] - 1.0, 4) if np.isfinite(sma.loc[asof]) else None,
            "chart": [{"date": d.strftime("%Y-%m-%d"), "price": fnum(a.loc[d], 2),
                       "sma": fnum(sma.loc[d], 2)} for d in prices.index[-chart_w:]],
        },
        "coins": coins,
        "book": book,
        "history": {"dates": [d.strftime("%Y-%m-%d") for d in hdates], "coins": hist_coins},
        "research": res,
    }

    OUT.write_text(json.dumps(_scrub(out), separators=(",", ":"), allow_nan=False))
    print(f"Saved {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")

    upsert_picks(HIST, {
        "date": out["generatedAt"][:10], "asOf": out["asOf"], "configHash": cfg_hash,
        "riskOn": out["regime"]["riskOn"],
        "markets": {"crypto": [{"ticker": b["ticker"], "name": b["name"],
                                "price": next((c["price"] for c in coins if c["ticker"] == b["ticker"]), None),
                                "weight": b["weight"]} for b in book]},
    })
    print(f"Upserted snapshot into {HIST.name}")


def _scrub(o):
    """Replace NaN/inf anywhere in the payload with None so the JSON is strict."""
    if isinstance(o, dict):
        return {k: _scrub(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_scrub(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


if __name__ == "__main__":
    main()
