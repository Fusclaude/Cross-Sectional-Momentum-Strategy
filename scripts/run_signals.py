"""
run_signals.py
──────────────
Computes, for every eligible ticker in each market:
  - all 8 raw factors (12-1, 9-1, 6-1, 3-1 skip-adjusted returns,
    and R² trend quality at 12/9/6/3 months)
  - current price
  - 12 months of MONTH-END prices with their actual dates
  - the full weekly price series (for the stock detail chart)

Unlike earlier versions, this does NOT pre-bake ranked picks for a fixed
list of scenarios. Instead it exports the raw per-ticker data and lets
the dashboard rank/score everything client-side — that's what makes the
weight sliders in the dashboard actually adjustable without re-running
this script. The Python side's job is just to be the trusted, scheduled
source of *facts* (factors + prices); the *ranking* is dashboard logic,
mirrored exactly from the validated JS engine.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

DATA_DIR = Path(__file__).parent.parent / "data"
WPM = 4.345  # weeks per month, matches the dashboard engine exactly


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


def compute_factors_at(prices_df: pd.DataFrame, pos: int, ipo_dates: dict, min_price: float = 1.0) -> dict:
    """All 8 factors for every eligible ticker at one point in time."""
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
        if np.isnan(p_now) or p_now < min_price:
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


def monthly_history(prices_df: pd.DataFrame, ticker: str, months: int = 12) -> list[dict]:
    """
    Resample this ticker's weekly series to MONTH-END prices for the
    trailing `months` months, with real calendar dates attached — this is
    what feeds the "every stock with monthly price + date + % change" list.
    """
    series = prices_df[ticker].dropna()
    if series.empty:
        return []
    monthly = series.resample("ME").last().dropna()
    monthly = monthly.tail(months + 1)  # +1 so we can compute the first % change
    out = []
    prev_val = None
    for dt, val in monthly.items():
        chg = None if prev_val is None or prev_val == 0 else (val / prev_val - 1)
        out.append({
            "date": dt.strftime("%Y-%m-%d"),
            "price": round(float(val), 4),
            "chg": round(float(chg), 4) if chg is not None else None,
        })
        prev_val = val
    return out[-months:] if len(out) > months else out


def run_market(market: str, prices_path: Path, min_price: float = 1.0) -> dict:
    with open(prices_path) as f:
        raw = json.load(f)

    dates = pd.to_datetime(raw["dates"])
    df = pd.DataFrame(raw["prices"], index=dates)
    names = raw["names"]
    sectors = raw.get("sectors", {})
    ipo_dates = raw.get("ipoDates", {})

    latest_pos = len(df) - 1
    max_lb = wk(12)
    if latest_pos < max_lb + wk(1) + 2:
        raise ValueError(f"{market}: not enough history yet ({latest_pos} weeks)")

    factors = compute_factors_at(df, latest_pos, ipo_dates, min_price=min_price)
    print(f"  {market}: {len(factors)} eligible tickers at {dates[-1].date()} (min price ${min_price})")

    stocks = []
    for t, f in factors.items():
        monthly = monthly_history(df, t, months=12)
        # full weekly series, capped to last ~2 years for chart detail
        # without bloating the payload indefinitely as history grows
        weekly_series = df[t].tail(104)
        chart = [
            {"date": dt.strftime("%Y-%m-%d"),
             "price": round(float(v), 4) if not np.isnan(v) else None}
            for dt, v in weekly_series.items()
        ]
        stocks.append({
            "ticker": t,
            "name": names.get(t, t),
            "sector": sectors.get(t, "—"),
            "price": f["price"],
            "factors": {k: round(v, 4) for k, v in f.items() if k != "price"},
            "monthly": monthly,
            "chart": chart,
        })

    return {
        "market": market,
        "asOf": dates[-1].strftime("%Y-%m-%d"),
        "universeSize": len(stocks),
        "universeUsedFallback": raw.get("universeUsedFallback", False),
        "stocks": stocks,
    }


# ─────────────────────────────────────────────────────────────────────────
#  REBALANCE STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────
#  Your backtests modelled a 28-day (4-week) rebalance cycle, NOT weekly
#  and NOT calendar-monthly. This job runs weekly so the *prices* on the
#  dashboard stay fresh, but the *portfolio* only rotates every 28 days --
#  matching the strategy you actually validated.
#
#  Between rebalances the dashboard still shows live signals (so you can
#  see what's brewing), but it clearly separates "what you hold" from
#  "what today's signal says", and won't tempt you into weekly trading
#  that the backtest never tested.
# ─────────────────────────────────────────────────────────────────────────

REBAL_DAYS = 28
TOP_N = 5

# Default strategy: pure 9-1 month momentum. Chosen because in the
# 2-to-15 month lookback sweep this window had the best Sharpe and the
# smallest drawdown on BOTH the S&P and ASX -- a cleaner risk-adjusted
# result than the 12-1 academic standard on this data.
DEFAULT_WEIGHTS = {
    "r121": 0.0, "r91": 1.0, "r61": 0.0, "r31": 0.0,
    "r2_12": 0.0, "r2_9": 0.0, "r2_6": 0.0, "r2_3": 0.0,
}


def rank_and_score(stocks: list[dict], weights: dict) -> dict:
    """Cross-sectional percentile-rank blend. Mirrors the dashboard's JS engine."""
    keys = [k for k, w in weights.items() if w > 0]
    n = len(stocks)
    ranks = {}
    for key in keys:
        vals = [s["factors"][key] for s in stocks]
        srt = sorted(vals)
        ranks[key] = {
            s["ticker"]: (sum(1 for x in srt if x < v) + 0.5) / n
            for s, v in zip(stocks, vals)
        }
    scores = {}
    for s in stocks:
        t = s["ticker"]
        scores[t] = sum(weights[k] * ranks[k][t] for k in keys) * 100
    return scores


def load_portfolio_state(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"markets": {}}


def process_rebalance(market: str, market_data: dict, state: dict, today: datetime) -> dict:
    """
    Decide whether a rebalance is DUE for this market, and if so rotate the
    portfolio. Returns the per-market state block (holdings + cycle info).
    """
    stocks = market_data["stocks"]
    scores = rank_and_score(stocks, DEFAULT_WEIGHTS)
    ranked = sorted(stocks, key=lambda s: -scores[s["ticker"]])
    target = ranked[:TOP_N]
    target_tickers = [s["ticker"] for s in target]

    prev = state.get("markets", {}).get(market)

    # First ever run for this market -> establish the initial portfolio
    if not prev or not prev.get("holdings"):
        return {
            "lastRebalance": today.strftime("%Y-%m-%d"),
            "nextRebalance": (today + timedelta(days=REBAL_DAYS)).strftime("%Y-%m-%d"),
            "holdings": [
                {"ticker": s["ticker"], "name": s["name"],
                 "entryPrice": s["price"], "entryDate": today.strftime("%Y-%m-%d")}
                for s in target
            ],
            "bought": target_tickers,
            "kept": [],
            "sold": [],
            "rebalancedThisRun": True,
            "cycleNumber": 1,
        }

    last_rebal = datetime.strptime(prev["lastRebalance"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days_since = (today - last_rebal).days

    # Not due yet -> hold. Carry the existing portfolio forward untouched.
    if days_since < REBAL_DAYS:
        return {
            **prev,
            "rebalancedThisRun": False,
            "daysSinceRebalance": days_since,
            "daysUntilRebalance": REBAL_DAYS - days_since,
        }

    # DUE -> rotate into the new target portfolio
    held_now = {h["ticker"] for h in prev["holdings"]}
    bought = [t for t in target_tickers if t not in held_now]
    kept = [t for t in target_tickers if t in held_now]
    sold = [t for t in held_now if t not in target_tickers]

    price_of = {s["ticker"]: s["price"] for s in stocks}
    prev_entry = {h["ticker"]: h for h in prev["holdings"]}

    new_holdings = []
    for s in target:
        t = s["ticker"]
        if t in kept:
            # carry original entry price/date so P&L is measured from first purchase
            new_holdings.append(prev_entry[t])
        else:
            new_holdings.append({
                "ticker": t, "name": s["name"],
                "entryPrice": s["price"], "entryDate": today.strftime("%Y-%m-%d"),
            })

    # Realised result on the names we just exited
    closed = []
    for t in sold:
        h = prev_entry[t]
        exit_price = price_of.get(t)
        pnl = None
        if exit_price and h.get("entryPrice"):
            pnl = round(exit_price / h["entryPrice"] - 1, 4)
        closed.append({
            "ticker": t, "name": h.get("name", t),
            "entryPrice": h.get("entryPrice"), "exitPrice": exit_price,
            "entryDate": h.get("entryDate"), "exitDate": today.strftime("%Y-%m-%d"),
            "pnl": pnl,
        })

    return {
        "lastRebalance": today.strftime("%Y-%m-%d"),
        "nextRebalance": (today + timedelta(days=REBAL_DAYS)).strftime("%Y-%m-%d"),
        "holdings": new_holdings,
        "bought": bought,
        "kept": kept,
        "sold": sold,
        "closed": closed,
        "rebalancedThisRun": True,
        "daysSinceRebalance": 0,
        "daysUntilRebalance": REBAL_DAYS,
        "cycleNumber": prev.get("cycleNumber", 1) + 1,
    }


def main():
    out = {"generatedAt": datetime.now(timezone.utc).isoformat() + "Z", "markets": {}}

    # ASX has many more legitimate small-cap stocks trading genuinely
    # below $1 AUD than the S&P does below $1 USD (where sub-$1 is more
    # often a sign of real distress) -- use a lower screen for ASX so
    # real constituents like A4N ($0.85), MI6, BC8 etc. aren't dropped.
    MIN_PRICE = {"sp500": 1.0, "asx300": 0.10}

    for market, fname in [("sp500", "sp500_prices.json"), ("asx300", "asx300_prices.json")]:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  Skipping {market}: {fname} not found (run fetch_prices.py first)")
            continue
        print(f"Computing signals for {market}...")
        out["markets"][market] = run_market(market, path, min_price=MIN_PRICE.get(market, 1.0))

    # ── Rebalance state machine ──────────────────────────────────────────
    state_path = DATA_DIR / "portfolio_state.json"
    state = load_portfolio_state(state_path)
    today = datetime.now(timezone.utc)

    print(f"\nRebalance check ({REBAL_DAYS}-day cycle, Top {TOP_N}, 9-1 momentum):")
    for market, market_data in out["markets"].items():
        block = process_rebalance(market, market_data, state, today)
        state.setdefault("markets", {})[market] = block
        if block["rebalancedThisRun"]:
            print(f"  {market}: REBALANCED (cycle {block['cycleNumber']}) "
                  f"-> bought {block['bought']}, sold {block.get('sold', [])}")
        else:
            print(f"  {market}: holding ({block['daysSinceRebalance']}d since last, "
                  f"{block['daysUntilRebalance']}d until next)")

    state["rebalDays"] = REBAL_DAYS
    state["topN"] = TOP_N
    state["strategy"] = "9-1 month momentum (pure)"
    state["updatedAt"] = out["generatedAt"]
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Saved {state_path}")

    # Embed portfolio state into the dashboard payload so it only has to
    # fetch a single file.
    out["portfolio"] = state

    out_path = DATA_DIR / "signals_latest.json"
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Saved {out_path}")

    # Append a history record ONLY on an actual rebalance -- so the history
    # page is a true record of trades, not a weekly noise log.
    if any(state["markets"][m]["rebalancedThisRun"] for m in state.get("markets", {})):
        history_path = DATA_DIR / "picks_history.jsonl"
        snapshot = {
            "date": out["generatedAt"][:10],
            "strategy": state["strategy"],
            "markets": {
                m: {
                    "holdings": [h["ticker"] for h in state["markets"][m]["holdings"]],
                    "bought": state["markets"][m].get("bought", []),
                    "sold": state["markets"][m].get("sold", []),
                    "closed": state["markets"][m].get("closed", []),
                }
                for m in state["markets"]
                if state["markets"][m]["rebalancedThisRun"]
            },
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
        print(f"Appended rebalance record to {history_path}")
    else:
        print("No rebalance this run -- history not appended.")


if __name__ == "__main__":
    main()
