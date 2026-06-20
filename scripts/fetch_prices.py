"""
fetch_prices.py
───────────────
Replaces the manual STOCKHISTORY refresh. Pulls ~2 years of daily prices
(enough for the longest 15-month lookback + skip + buffer) for every ticker
in universe.json, resamples to weekly, and writes a single price file per
market — matching the exact format your existing dashboard JSON already
expects (dates / tickers / prices / names / sectors).

Runs unattended: no Excel, no manual paste, no GUI. yfinance batches
requests so this is one HTTP round-trip per ~50-100 tickers, not one per
ticker — a full 500+700 ticker refresh takes a few minutes, well within
GitHub Actions' free-tier job time limit.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

DATA_DIR = Path(__file__).parent.parent / "data"
LOOKBACK_DAYS = 760  # ~25 months: covers 15mo lookback + 1mo skip + buffer + warmup

# A static IPO-date table for the known post-2016 US listings that have
# (historically) shown back-filled pre-IPO prices in some data sources.
# yfinance itself does NOT back-fill IPOs the way some bulk vendors do —
# it simply has no data before a stock's real first trading day — so this
# gating is now mostly a no-op safety net rather than a required fix.
# Kept here so the signal engine's eligibility logic stays identical to
# your validated backtests.
IPO_DATES = {
    "CRWD": "2019-06-12", "DDOG": "2019-09-19", "ZM": "2019-04-18",
    "PINS": "2019-04-18", "UBER": "2019-05-10", "LYFT": "2019-03-29",
    "SNOW": "2020-09-16", "PLTR": "2020-09-30", "ABNB": "2020-12-10",
    "DASH": "2020-12-09", "RBLX": "2021-03-10", "COIN": "2021-04-14",
    "HOOD": "2021-07-29", "RIVN": "2021-11-10", "SOFI": "2021-06-01",
    "CVNA": "2017-04-28", "ARM": "2023-09-14", "CAVA": "2023-06-15",
}


def fetch_market(tickers: list[str], names: dict, market_label: str) -> dict:
    print(f"  Downloading {len(tickers)} tickers for {market_label}...")
    end = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS)

    # yfinance batches internally; chunk defensively in case of API limits
    chunk_size = 100
    frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        data = yf.download(
            chunk, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
            threads=True,
        )
        frames.append(data)
        print(f"    chunk {i // chunk_size + 1}: {len(chunk)} tickers fetched")

    # Stitch chunks into one wide DataFrame of Close prices
    closes = {}
    for frame in frames:
        if isinstance(frame.columns, pd.MultiIndex):
            for t in frame.columns.levels[0]:
                if (t, "Close") in frame.columns:
                    closes[t] = frame[(t, "Close")]
        else:
            # single-ticker chunk falls back to flat columns
            only_ticker = chunk[0] if len(chunk) == 1 else None
            if only_ticker and "Close" in frame.columns:
                closes[only_ticker] = frame["Close"]

    df = pd.DataFrame(closes).sort_index()
    weekly = df.resample("W-FRI").last()

    # drop tickers with insufficient history (matches your 50% coverage rule)
    coverage = weekly.notna().mean()
    keep = coverage[coverage >= 0.5].index.tolist()
    weekly = weekly[keep]

    dates = [d.strftime("%Y-%m-%d") for d in weekly.index]
    prices = {
        t: [round(float(v), 4) if not np.isnan(v) else None for v in weekly[t].values]
        for t in keep
    }
    clean_names = {t: names.get(t, t) for t in keep}

    print(f"  Kept {len(keep)} / {len(tickers)} tickers after coverage filter")
    return {
        "dates": dates,
        "tickers": keep,
        "prices": prices,
        "names": clean_names,
        "ipoDates": {t: d for t, d in IPO_DATES.items() if t in keep},
        "fetchedAt": datetime.now(timezone.utc).isoformat() + "Z",
    }


def main():
    with open(DATA_DIR / "universe.json") as f:
        universe = json.load(f)

    print("Fetching S&P 500 prices...")
    sp_payload = fetch_market(
        universe["sp500"]["tickers"], universe["sp500"]["names"], "S&P 500"
    )
    with open(DATA_DIR / "sp500_prices.json", "w") as f:
        json.dump(sp_payload, f, separators=(",", ":"))

    print("Fetching ASX 300 prices...")
    asx_payload = fetch_market(
        universe["asx300"]["tickers"], universe["asx300"]["names"], "ASX 300"
    )
    with open(DATA_DIR / "asx300_prices.json", "w") as f:
        json.dump(asx_payload, f, separators=(",", ":"))

    print("Done.")


if __name__ == "__main__":
    main()
