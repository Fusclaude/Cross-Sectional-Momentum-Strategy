"""
fetch_prices.py  (v2)
─────────────────────
Changes from v1, and why each one matters:

1. HISTORY: 760 days -> 3650 days (config-driven).
   v1 pulled just enough history to compute a 12-1 signal today. That left
   roughly 60 usable weekly cross-sections for evaluation, and 60 observations
   cannot distinguish a real information coefficient from noise — the standard
   error on an IC estimate over 60 periods is around 0.13, which is wider than
   any equity factor's true IC. Ten years costs nothing extra from the same
   API call and is the difference between "the backtest looks good" and "the
   signal is measurable".

2. VOLUME: now fetched alongside Close.
   Without volume there is no liquidity screen, no ADV participation cap, and
   no Amihud illiquidity measure — which means a backtest is free to "buy" a
   microcap at a price nobody could transact in size. Nearly all implausibly
   good small-cap backtests die on this one constraint.

3. BENCHMARK + RISK-FREE: index and T-bill series fetched explicitly.
   An equal-weight universe average is a poor market proxy (it is implicitly
   a small-cap tilt). Beta, alpha, Sharpe and information ratio all need the
   real thing.

4. QUALITY GATE: writes a machine-readable report and exits non-zero on
   failure, so a bad pull fails the CI job instead of quietly publishing a
   ranking built on stale prices.

5. RETRIES with backoff, and an atomic write, so a half-written JSON can never
   be committed.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Retained from v1: seasoning gate for recent listings. yfinance does not
# back-fill pre-IPO prices, so this is a safety net rather than a fix, but it
# keeps eligibility identical to the validated backtests.
IPO_DATES = {
    "CRWD": "2019-06-12", "DDOG": "2019-09-19", "ZM": "2019-04-18",
    "PINS": "2019-04-18", "UBER": "2019-05-10", "LYFT": "2019-03-29",
    "SNOW": "2020-09-16", "PLTR": "2020-09-30", "ABNB": "2020-12-10",
    "DASH": "2020-12-09", "RBLX": "2021-03-10", "COIN": "2021-04-14",
    "HOOD": "2021-07-29", "RIVN": "2021-11-10", "SOFI": "2021-06-01",
    "CVNA": "2017-04-28", "ARM": "2023-09-14", "CAVA": "2023-06-15",
}


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _download(tickers: list[str], start, end, attempts: int = 3) -> pd.DataFrame:
    """yfinance batch download with exponential backoff."""
    last = None
    for i in range(attempts):
        try:
            df = yf.download(tickers, start=start, end=end, interval="1d",
                             auto_adjust=True, progress=False, group_by="ticker",
                             threads=True)
            if df is not None and len(df):
                return df
            last = ValueError("empty frame")
        except Exception as e:  # transient API/network failure
            last = e
        wait = 2 ** i * 5
        print(f"    retry {i + 1}/{attempts} in {wait}s ({last})")
        time.sleep(wait)
    raise RuntimeError(f"download failed after {attempts} attempts: {last}")


def _extract(frames: list[pd.DataFrame], chunks: list[list[str]], field: str) -> pd.DataFrame:
    """Pull one OHLCV field out of the grouped-by-ticker frames."""
    cols = {}
    for frame, chunk in zip(frames, chunks):
        if isinstance(frame.columns, pd.MultiIndex):
            for t in frame.columns.get_level_values(0).unique():
                if (t, field) in frame.columns:
                    cols[t] = frame[(t, field)]
        elif len(chunk) == 1 and field in frame.columns:
            cols[chunk[0]] = frame[field]
    return pd.DataFrame(cols).sort_index()


def fetch_market(tickers: list[str], names: dict, sectors: dict,
                 label: str, cfg: dict) -> dict:
    dcfg = cfg["data"]
    end = datetime.today()
    start = end - timedelta(days=dcfg["lookback_days"])
    print(f"  {label}: {len(tickers)} tickers, {dcfg['lookback_days']}d history")

    chunk_size = 100
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    frames = []
    for i, chunk in enumerate(chunks):
        frames.append(_download(chunk, start, end))
        print(f"    chunk {i + 1}/{len(chunks)}")

    daily_px = _extract(frames, chunks, "Close")
    daily_vol = _extract(frames, chunks, "Volume")

    # Dollar volume must be computed DAILY then resampled. Multiplying a
    # weekly price by a weekly volume sum is a different (and wrong) quantity.
    daily_dv = (daily_px * daily_vol).reindex(columns=daily_px.columns)

    rule = dcfg["resample"]
    weekly = daily_px.resample(rule).last()
    weekly_dv = daily_dv.resample(rule).mean()          # avg daily $ volume in week
    median_dv = daily_dv.tail(60).median()              # 60-trading-day median

    coverage = weekly.notna().mean()
    keep = coverage[coverage >= dcfg["min_coverage"]].index.tolist()
    weekly, weekly_dv = weekly[keep], weekly_dv.reindex(columns=keep)
    print(f"    kept {len(keep)}/{len(tickers)} after coverage filter")

    dates = [d.strftime("%Y-%m-%d") for d in weekly.index]

    def pack(df: pd.DataFrame) -> dict:
        return {t: [None if not np.isfinite(v) else round(float(v), 4)
                    for v in df[t].to_numpy()] for t in df.columns}

    return {
        "dates": dates,
        "tickers": keep,
        "prices": pack(weekly),
        "dollarVolume": pack(weekly_dv),
        "medianDollarVolume60d": {t: (None if not np.isfinite(median_dv.get(t, np.nan))
                                      else round(float(median_dv[t]), 0)) for t in keep},
        "names": {t: names.get(t, t) for t in keep},
        "sectors": {t: sectors.get(t, "—") for t in keep},
        "ipoDates": {t: d for t, d in IPO_DATES.items() if t in keep},
        "fetchedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def fetch_series(symbol: str, start, end) -> dict | None:
    """Single auxiliary series (benchmark index or T-bill yield)."""
    if not symbol:
        return None
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        col = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        wk = col.resample("W-FRI").last().dropna()
        return {"symbol": symbol,
                "dates": [d.strftime("%Y-%m-%d") for d in wk.index],
                "values": [round(float(v), 6) for v in wk.to_numpy().ravel()]}
    except Exception as e:
        print(f"    WARNING: {symbol} fetch failed ({e})")
        return None


def quality_gate(payload: dict, label: str) -> list[str]:
    """
    Hard checks. Returning a non-empty list fails the run. These exist because
    the failure mode of bad market data is not a crash — it is a perfectly
    plausible-looking top-5 list built on a stale or unsplit price series.
    """
    problems = []
    px = pd.DataFrame(payload["prices"], index=pd.to_datetime(payload["dates"]))

    if len(px) < 60:
        problems.append(f"{label}: only {len(px)} weekly bars; need >=60")

    missing = px.isna().mean().mean()
    if missing > 0.15:
        problems.append(f"{label}: {missing:.1%} missing values (limit 15%)")

    rets = px.pct_change()
    # A -50% weekly move that immediately reverses is the classic signature of
    # an unadjusted stock split, not a real price move.
    split_like = ((rets < -0.45) & (rets.shift(-1) > 0.8)).sum().sum()
    if split_like > 0:
        problems.append(f"{label}: {split_like} suspected unadjusted splits")

    # Every price identical for 4+ weeks means a dead feed, not a quiet stock.
    stale = ((rets.abs() < 1e-12).rolling(4).sum() >= 4).any()
    n_stale = int(stale.sum())
    if n_stale > len(px.columns) * 0.05:
        problems.append(f"{label}: {n_stale} tickers with 4+ weeks of identical prices")

    last_date = px.index[-1]
    age_days = (pd.Timestamp.today().normalize() - last_date.normalize()).days
    if age_days > 10:
        problems.append(f"{label}: latest bar is {age_days}d old ({last_date.date()})")

    return problems


def atomic_write(path: Path, payload: dict) -> None:
    """Write to a temp file and rename. A killed job cannot leave a half-file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def main() -> None:
    cfg = load_config()
    with open(DATA_DIR / "universe.json") as f:
        universe = json.load(f)

    end = datetime.today()
    start = end - timedelta(days=cfg["data"]["lookback_days"])
    all_problems: list[str] = []

    for market, label in [("sp500", "S&P 500"), ("asx300", "ASX 300")]:
        u = universe[market]
        payload = fetch_market(u["tickers"], u["names"], u.get("sectors", {}), label, cfg)
        payload["universeUsedFallback"] = u.get("usedFallback", False)
        payload["benchmark"] = fetch_series(cfg["data"]["benchmark"].get(market), start, end)
        payload["riskFree"] = fetch_series(cfg["data"]["risk_free"].get(market), start, end)

        problems = quality_gate(payload, label)
        payload["qualityProblems"] = problems
        all_problems += problems
        atomic_write(DATA_DIR / f"{market}_prices.json", payload)
        print(f"  wrote {market}_prices.json"
              + (f"  [{len(problems)} QA problems]" if problems else "  [QA clean]"))

    if all_problems:
        print("\nDATA QUALITY GATE FAILED:")
        for p in all_problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nDone. QA clean.")


if __name__ == "__main__":
    main()
