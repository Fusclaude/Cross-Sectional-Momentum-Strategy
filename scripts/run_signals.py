"""
run_signals.py
──────────────
Python port of the scenario tool's JS engine — same 8 factors, same
cross-sectional ranking, same eligibility rules — so the numbers this
produces match what you'd see if you ran the interactive HTML tool by
hand. This is what makes the dashboard "current" without you opening
Excel: this script runs on a schedule, computes the latest rankings,
and writes a small JSON file the dashboard reads on load.

Produces, per market, per scenario:
  - current Top-5/10/20 picks with full factor breakdown
  - each ticker's rank under each scenario (for the "view a ticker's
    rank across blends" feature)
  - trailing price series per ticker (for the trend sparkline)
  - last N months of historical picks (for "did this call work out")
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

DATA_DIR = Path(__file__).parent.parent / "data"
WPM = 4.345  # weeks per month, matches the JS engine exactly

# Scenarios are defined once here, in Python, and mirrored in the
# dashboard purely for display — the math always happens here so
# there is exactly one source of truth.
SCENARIOS = {
    "pure_momentum": {
        "label": "Pure Momentum",
        "weights": {"r121": 0.30, "r91": 0.20, "r61": 0.15, "r31": 0.15,
                    "r2_12": 0.05, "r2_9": 0.05, "r2_6": 0.05, "r2_3": 0.05},
    },
    "trend_quality": {
        "label": "Trend Quality",
        "weights": {"r121": 0.12, "r91": 0.08, "r61": 0.08, "r31": 0.05,
                    "r2_12": 0.25, "r2_9": 0.20, "r2_6": 0.12, "r2_3": 0.10},
    },
    "balanced": {
        "label": "Balanced",
        "weights": {"r121": 0.14, "r91": 0.13, "r61": 0.12, "r31": 0.11,
                    "r2_12": 0.13, "r2_9": 0.13, "r2_6": 0.12, "r2_3": 0.12},
    },
    "nine_month_pure": {
        "label": "9-Month Pure (your tested best-Sharpe signal)",
        "weights": {"r121": 0, "r91": 1.0, "r61": 0, "r31": 0,
                    "r2_12": 0, "r2_9": 0, "r2_6": 0, "r2_3": 0},
    },
}

TOPN_OPTIONS = [5, 10, 20]
REBAL_DAYS = 28
TCOST_BPS = 20


def wk(months: float) -> int:
    return round(months * WPM)


def r2_trend(prices: np.ndarray) -> float | None:
    valid = prices[~np.isnan(prices)]
    if len(valid) < len(prices) * 0.6 or len(valid) < 6:
        return None
    y = np.log(valid)
    x = np.arange(len(y))
    if np.std(y) == 0:
        return None
    coef = np.polyfit(x, y, 1)
    yhat = np.polyval(coef, x)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return None
    return max(0.0, 1 - ss_res / ss_tot)


def eligible(ticker: str, date: pd.Timestamp, ipo_dates: dict) -> bool:
    ipo = ipo_dates.get(ticker)
    if ipo:
        ipo_dt = pd.Timestamp(ipo)
        return date >= ipo_dt + timedelta(days=365)
    return True


def compute_factors_at(prices_df: pd.DataFrame, pos: int, ipo_dates: dict) -> dict:
    """Mirrors the JS computeBacktest's per-rebalance factor computation."""
    LB = {"r121": wk(12), "r91": wk(9), "r61": wk(6), "r31": wk(3)}
    SKIP = wk(1)
    R2W = {"r2_12": wk(12), "r2_9": wk(9), "r2_6": wk(6), "r2_3": wk(3)}

    date = prices_df.index[pos]
    factors = {}
    for t in prices_df.columns:
        if not eligible(t, date, ipo_dates):
            continue
        arr = prices_df[t].values
        p_now = arr[pos]
        if np.isnan(p_now) or p_now < 1:
            continue
        try:
            p121b, p91b, p61b, p31b = (
                arr[pos - LB["r121"]], arr[pos - LB["r91"]],
                arr[pos - LB["r61"]], arr[pos - LB["r31"]],
            )
            p_skip = arr[pos - SKIP]
        except IndexError:
            continue
        if any(np.isnan(v) or v <= 0 for v in (p121b, p91b, p61b, p31b, p_skip)):
            continue

        r121 = p_skip / p121b - 1
        r91 = p_skip / p91b - 1
        r61 = p_skip / p61b - 1
        r31 = p_skip / p31b - 1

        r2_vals = {}
        ok = True
        for key, w in R2W.items():
            if pos - w < 0:
                ok = False
                break
            window = arr[pos - w: pos + 1]
            r2 = r2_trend(window)
            if r2 is None:
                ok = False
                break
            r2_vals[key] = r2
        if not ok:
            continue

        factors[t] = dict(r121=r121, r91=r91, r61=r61, r31=r31,
                           price=float(p_now), **r2_vals)
    return factors


def rank_and_score(factors: dict, weights: dict) -> dict:
    names = list(factors.keys())
    ranks = {}
    for key in weights:
        vals = np.array([factors[t][key] for t in names])
        order = vals.argsort()
        ranks_arr = np.empty_like(order, dtype=float)
        # average-rank percentile (matches the JS tie handling closely enough)
        sorted_vals = np.sort(vals)
        for i, v in enumerate(vals):
            cnt = np.sum(sorted_vals < v)
            ranks_arr[i] = (cnt + 0.5) / len(vals)
        ranks[key] = dict(zip(names, ranks_arr))

    scores = {}
    for t in names:
        scores[t] = sum(weights[k] * ranks[k][t] for k in weights) * 100
    return scores


def run_market(market: str, prices_path: Path) -> dict:
    with open(prices_path) as f:
        raw = json.load(f)

    dates = pd.to_datetime(raw["dates"])
    df = pd.DataFrame(raw["prices"], index=dates)
    names = raw["names"]
    ipo_dates = raw.get("ipoDates", {})

    latest_pos = len(df) - 1
    max_lb = wk(12)
    if latest_pos < max_lb + wk(1) + 2:
        raise ValueError(f"{market}: not enough history yet ({latest_pos} weeks)")

    factors = compute_factors_at(df, latest_pos, ipo_dates)
    print(f"  {market}: {len(factors)} eligible tickers at {dates[-1].date()}")

    result = {
        "market": market,
        "asOf": dates[-1].strftime("%Y-%m-%d"),
        "universeSize": len(factors),
        "universeUsedFallback": raw.get("universeUsedFallback", False),
        "scenarios": {},
    }

    for scen_id, scen in SCENARIOS.items():
        scores = rank_and_score(factors, scen["weights"])
        ranked = sorted(scores.items(), key=lambda x: -x[1])

        picks = {}
        for n in TOPN_OPTIONS:
            top = ranked[:n]
            picks[str(n)] = [
                {
                    "ticker": t,
                    "name": names.get(t, t),
                    "score": round(s, 1),
                    "rank": i + 1,
                    "factors": {k: round(v, 4) for k, v in factors[t].items() if k != "price"},
                    "price": factors[t]["price"],
                    "spark": [
                        round(v, 4) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
                        for v in df[t].values[-26:]  # last 6 months weekly, for sparkline
                    ],
                }
                for i, (t, s) in enumerate(top)
            ]

        # full rank table (every eligible ticker, for "view ticker rank across blends")
        full_ranks = {t: i + 1 for i, (t, s) in enumerate(ranked)}

        result["scenarios"][scen_id] = {
            "label": scen["label"],
            "weights": scen["weights"],
            "picks": picks,
            "fullRanks": full_ranks,
        }

    return result


def main():
    out = {"generatedAt": datetime.now(timezone.utc).isoformat() + "Z", "markets": {}}

    for market, fname in [("sp500", "sp500_prices.json"), ("asx300", "asx300_prices.json")]:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  Skipping {market}: {fname} not found (run fetch_prices.py first)")
            continue
        print(f"Computing signals for {market}...")
        out["markets"][market] = run_market(market, path)

    out_path = DATA_DIR / "signals_latest.json"
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Saved {out_path}")

    # Append a lightweight history record (just the Top-5 pure-momentum
    # picks per market) so the dashboard can show "did last month's call
    # work out" without keeping every scenario's full history forever.
    history_path = DATA_DIR / "picks_history.jsonl"
    snapshot = {
        "date": out["generatedAt"][:10],
        "markets": {
            m: out["markets"][m]["scenarios"]["pure_momentum"]["picks"]["5"]
            for m in out["markets"]
        },
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
    print(f"Appended snapshot to {history_path}")


if __name__ == "__main__":
    main()
