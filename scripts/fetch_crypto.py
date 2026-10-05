"""
fetch_crypto.py
───────────────
Pulls daily crypto prices and volume from Yahoo (yfinance) for every coin in
config_crypto.yaml's candidate list, converts to CLOSED weekly bars and writes
data/crypto_prices.json.

Differences from the equity fetch, each deliberate:

  * Weeks end Sunday UTC (crypto has no Friday close) and the in-progress
    day/week is dropped, so the run time never changes the numbers.
  * Volume units are auto-detected (Yahoo reports crypto volume in USD,
    equities in shares) and recorded in the output.
  * Missing tickers are expected and NOT fatal. A survivorship-aware
    candidate list contains dead and renamed coins by design.
  * No split check. Crypto has genuine -90% weeks; those are reported as
    warnings so they can be eyeballed, but they are real prices.

Exit code is non-zero only for problems that would corrupt today's ranking:
stale data, a required coin missing, or too thin an investable universe.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crypto_engine as ce  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT = DATA_DIR / "crypto_prices.json"


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config_crypto.yaml").read_text())


def _download(tickers: list[str], start: str, attempts: int = 3) -> pd.DataFrame:
    import yfinance as yf
    last = None
    for i in range(attempts):
        try:
            df = yf.download(tickers, start=start, interval="1d", auto_adjust=True,
                             progress=False, group_by="ticker", threads=True)
            if df is not None and len(df):
                return df
            last = ValueError("empty frame")
        except Exception as e:  # transient API/network failure
            last = e
        wait = 2 ** i * 5
        print(f"    retry {i + 1}/{attempts} in {wait}s ({last})")
        time.sleep(wait)
    raise RuntimeError(f"download failed after {attempts} attempts: {last}")


def _field(frame: pd.DataFrame, chunk: list[str], field: str) -> dict:
    cols = {}
    if isinstance(frame.columns, pd.MultiIndex):
        for t in frame.columns.get_level_values(0).unique():
            if (t, field) in frame.columns:
                cols[t] = frame[(t, field)]
    elif len(chunk) == 1 and field in frame.columns:
        cols[chunk[0]] = frame[field]
    return cols


def quality_gate(weekly: pd.DataFrame, dv: pd.DataFrame, cfg: dict,
                 now: pd.Timestamp) -> tuple[list[str], list[str]]:
    """Returns (failures, warnings)."""
    q = cfg["quality"]
    fails, warns = [], []
    if weekly.empty:
        return ["no weekly bars at all"], warns

    last = weekly.index[-1]
    age = (now.tz_localize(None).normalize() - last).days
    if age > q["max_staleness_days"]:
        fails.append(f"latest closed week {last.date()} is {age} days old")

    for t in q["required"]:
        if t not in weekly.columns or not np.isfinite(weekly[t].iloc[-1]):
            fails.append(f"required coin {t} has no price for {last.date()}")

    # Feeds that stopped updating: same price for the last 4 closed weeks.
    tail = weekly.tail(4)
    flat = [c for c in weekly.columns
            if tail[c].notna().all() and tail[c].nunique() == 1]
    if flat:
        warns.append(f"flat for 4 weeks (dead feed?): {', '.join(sorted(flat))}")

    r = weekly.pct_change(fill_method=None)
    lim = float(q["extreme_week_warn"])
    big = r.where((r > lim) | (r < -lim / (1 + lim)))
    hits = big.stack().dropna()
    if len(hits):
        recent = hits[hits.index.get_level_values(0) >= weekly.index[-1] - pd.Timedelta(weeks=8)]
        if len(recent):
            warns.append("extreme weekly moves in the last 8 weeks: " + ", ".join(
                f"{t} {d.date()} {v:+.0%}" for (d, t), v in recent.items()))

    panels = ce.factor_panel(weekly, cfg)
    uni = ce.universe_mask(weekly, dv, cfg, factor_ok=ce.factors_complete(panels))
    n_now = int(uni.iloc[-1].sum())
    if n_now < q["min_universe"]:
        fails.append(f"only {n_now} investable coins at {last.date()} "
                     f"(min {q['min_universe']})")
    return fails, warns


def main() -> None:
    cfg = load_config()
    cands: dict = cfg["universe"]["candidates"]
    tickers = list(cands)
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    print(f"Crypto: {len(tickers)} candidates from {cfg['data']['start']}")

    px_cols, vol_cols = {}, {}
    chunks = [tickers[i:i + 50] for i in range(0, len(tickers), 50)]
    for i, chunk in enumerate(chunks):
        frame = _download(chunk, cfg["data"]["start"])
        px_cols.update(_field(frame, chunk, "Close"))
        vol_cols.update(_field(frame, chunk, "Volume"))
        print(f"  chunk {i + 1}/{len(chunks)}")

    daily_px = pd.DataFrame(px_cols).sort_index()
    daily_vol = pd.DataFrame(vol_cols).reindex(columns=daily_px.columns).sort_index()
    daily_px.index = pd.DatetimeIndex(daily_px.index).tz_localize(None)
    daily_vol.index = pd.DatetimeIndex(daily_vol.index).tz_localize(None)

    missing = sorted(t for t in tickers
                     if t not in daily_px.columns or daily_px[t].notna().sum() == 0)
    daily_px = daily_px.drop(columns=missing, errors="ignore")
    daily_vol = daily_vol.drop(columns=missing, errors="ignore")
    if missing:
        print(f"  {len(missing)} candidates have no Yahoo data: {', '.join(missing)}")

    units = cfg["data"].get("volume_units", "auto")
    if units == "auto":
        units = ce.detect_volume_units(daily_vol.get("BTC-USD", pd.Series(dtype=float)),
                                       daily_px.get("BTC-USD", pd.Series(dtype=float)))
    daily_dv = daily_vol if units == "usd" else daily_vol * daily_px
    print(f"  volume units: {units}")

    rule = cfg["data"]["resample"]
    weekly = ce.complete_weekly(daily_px, rule, now, how="last")
    weekly_dv = ce.complete_weekly(daily_dv, rule, now, how="mean").reindex(weekly.index)
    # A price that is zero or negative is a feed error, never a real quote.
    weekly = weekly.where(weekly > 0)

    fails, warns = quality_gate(weekly, weekly_dv, cfg, now)
    for w in warns:
        print("  WARN:", w)

    def col(df: pd.DataFrame, t: str) -> list:
        return [None if not np.isfinite(v) else float(f"{v:.8g}") for v in df[t].to_numpy()]

    payload = {
        "generatedAt": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resample": rule,
        "volumeUnits": units,
        "dates": [d.strftime("%Y-%m-%d") for d in weekly.index],
        "prices": {t: col(weekly, t) for t in weekly.columns},
        "dollarVolume": {t: col(weekly_dv, t) for t in weekly.columns},
        "names": {t: cands[t].get("name", t) for t in weekly.columns},
        "categories": {t: cands[t].get("category", "—") for t in weekly.columns},
        "missingTickers": missing,
        "qualityProblems": fails,
        "qualityWarnings": warns,
    }
    DATA_DIR.mkdir(exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    tmp.replace(OUT)
    print(f"Saved {OUT} ({len(weekly)} weeks x {weekly.shape[1]} coins, "
          f"last week {weekly.index[-1].date() if len(weekly) else '—'})")

    if fails:
        print("QUALITY GATE FAILED:")
        for f in fails:
            print("  -", f)
        if os.environ.get("QA_SOFT_FAIL") != "1":
            sys.exit(1)


if __name__ == "__main__":
    main()
