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


def main():
    out = {"generatedAt": datetime.now(timezone.utc).isoformat() + "Z", "markets": {}}

    # ASX has many more legitimate small-cap stocks trading genuinely
    # below $1 AUD than the S&P does below $1 USD (where sub-$1 is more
    # often a sign of real distress) -- use a lower screen for ASX so
    # real constituents like A4N ($0.85), MI6, BC8 etc. aren't dropped.
    MIN_PRICE = {"sp500": 0.10, "asx300": 0.05}

    for market, fname in [("sp500", "sp500_prices.json"), ("asx300", "asx300_prices.json")]:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  Skipping {market}: {fname} not found (run fetch_prices.py first)")
            continue
        print(f"Computing signals for {market}...")
        out["markets"][market] = run_market(market, path, min_price=MIN_PRICE.get(market, 1.0))

    out_path = DATA_DIR / "signals_latest.json"
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Saved {out_path}")

    # Lightweight history snapshot: top 5 by default pure-12-1 momentum,
    # just for the "did this call work out" history page. The dashboard
    # itself can re-rank with any weights on the full stock list above;
    # this snapshot is only a fixed reference point over time.
    history_path = DATA_DIR / "picks_history.jsonl"
    snapshot = {"date": out["generatedAt"][:10], "markets": {}}
    for m, market_data in out["markets"].items():
        ranked = sorted(market_data["stocks"], key=lambda s: -s["factors"]["r121"])[:5]
        snapshot["markets"][m] = [
            {"ticker": s["ticker"], "name": s["name"], "price": s["price"]}
            for s in ranked
        ]
    with open(history_path, "a") as f:
        f.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
    print(f"Appended snapshot to {history_path}")


if __name__ == "__main__":
    main()
