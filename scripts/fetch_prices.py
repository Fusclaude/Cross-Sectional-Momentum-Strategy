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

    # ── Coverage filter ───────────────────────────────────────────────────
    # Measured SINCE EACH TICKER'S FIRST OBSERVATION, not against the whole
    # window. Measuring against the window silently deletes every recent
    # listing: with a 10-year fetch, Sandisk (spun out of Western Digital in
    # Feb 2025) has ~14% coverage and looks identical to a broken feed, so it
    # was dropped — while being the best-performing stock in the index.
    #
    # This bug did not exist at the old 760-day window, where the same stock
    # had 67% coverage. Extending history made the filter strictly harsher on
    # new listings, which is the opposite of the intent, and it introduced a
    # bias precisely against the high-momentum names this strategy exists to
    # find.
    #
    # Two separate conditions now, because they ask different questions:
    #   coverage  — is the feed intact over the period it has been listed?
    #   min bars  — is there enough history to compute a 12-1 signal at all?
    first_obs = weekly.notna().idxmax()
    n_obs = weekly.notna().sum()
    bars_since_listing = pd.Series(
        {t: int((weekly.index >= first_obs[t]).sum()) if n_obs[t] > 0 else 0
         for t in weekly.columns})
    coverage = (n_obs / bars_since_listing.replace(0, np.nan)).fillna(0.0)
    min_bars = dcfg.get("min_history_weeks", 60)

    keep = [t for t in weekly.columns
            if coverage[t] >= dcfg["min_coverage"] and n_obs[t] >= min_bars]
    dropped_short = [t for t in weekly.columns
                     if n_obs[t] < min_bars and coverage[t] >= dcfg["min_coverage"]]
    dropped_gappy = [t for t in weekly.columns if coverage[t] < dcfg["min_coverage"]]

    weekly, weekly_dv = weekly[keep], weekly_dv.reindex(columns=keep)
    print(f"    kept {len(keep)}/{len(tickers)}  "
          f"(dropped {len(dropped_gappy)} gappy, {len(dropped_short)} too short)")
    if dropped_short:
        print(f"      too short (<{min_bars}w, likely recent listings): "
              f"{', '.join(dropped_short[:10])}")
    if dropped_gappy:
        print(f"      gappy feed: {', '.join(dropped_gappy[:10])}")

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


def quality_gate(payload: dict, label: str, qcfg: dict,
                 min_price: float = 0.0) -> tuple[list[str], list[str]]:
    """
    Returns (failures, warnings).

    The first version of this function failed the run on ANY suspected split
    and on flat prices affecting >5% of tickers. Both thresholds were tuned
    against a 2-year panel, then applied unchanged to a 10-year one. Raw counts
    scale with history length, so the check became strictly more likely to fire
    the more data you added — which is backwards.

    Worse, both were flagging normal market behaviour:
      * ASX small caps enter trading halts and suspensions routinely. The last
        traded price legitimately repeats for weeks. That is valid data.
      * A large drop immediately reversed is the signature of an unadjusted
        split, but also of a real crash-and-bounce in a microcap.

    The gate now fails only on evidence worth acting on, and warns visibly on
    everything else. A gate that cries wolf gets switched off, and a gate that
    is switched off protects nothing.

    The organising principle is RECENCY: today's ranking is built from today's
    prices. A stock halted for six weeks in 2019 affects factor values around
    2019 and nothing you would trade on now. A feed that died three months ago
    is corrupting the live cross-section.
    """
    failures, warnings = [], []
    px = pd.DataFrame(payload["prices"], index=pd.to_datetime(payload["dates"]))

    # Tickers with known, reviewed, benign data quirks. Listing one here is an
    # explicit decision recorded in config with a reason — not the same thing
    # as turning the check off. The gate still runs on every other name, and
    # the ignored ones are reported so they stay visible.
    ignored = set(qcfg.get("ignore_tickers", []) or [])
    if ignored:
        present = [t for t in px.columns if t in ignored]
        if present:
            warnings.append(f"{label}: ignoring {len(present)} ticker(s) by config: "
                            f"{', '.join(present)}")
        px = px.drop(columns=present)

    n_tick = max(px.shape[1], 1)

    if len(px) < qcfg["min_weekly_bars"]:
        failures.append(f"{label}: only {len(px)} weekly bars; "
                        f"need >= {qcfg['min_weekly_bars']}")

    missing = px.isna().mean().mean()
    if missing > qcfg["max_missing_pct"]:
        failures.append(f"{label}: {missing:.1%} missing values "
                        f"(limit {qcfg['max_missing_pct']:.0%})")

    rets = px.pct_change()

    # ── Unadjusted splits: only a CLEAN ratio is evidence ────────────────
    # A true 2:1 split drops almost exactly -50% and recovers almost exactly
    # +100%. A real crash rarely retraces that precisely, so a ratio of 1.83
    # is a genuine move and 2.01 is a data error.
    #
    # TWO GUARDS, both learned from a false positive on LOT.AX:
    #
    # 1. Only inspect bars where the price clears the universe's min_price
    #    screen. A stock trading below the screen is not in the investable
    #    cross-section on that date, so a data error there cannot affect any
    #    ranking this system produces. Gating on it is noise.
    #
    # 2. Tick quantisation manufactures exact integer ratios in cheap stocks.
    #    The ASX quotes sub-$0.10 names in $0.001 increments, so a stock
    #    oscillating between $0.005 and $0.010 produces a perfect 2:1 ratio
    #    from a single tick. Require the price to be far enough above the tick
    #    that a 50% move cannot be a handful of ticks.
    floor = max(qcfg.get("split_check_min_price", 0.0), min_price)
    tick = qcfg.get("tick_size", 0.001)
    min_ticks = qcfg.get("split_check_min_ticks", 40)

    hit = ((rets < -0.45) & (rets.shift(-1) > 0.8)).fillna(False)
    clean, messy, skipped = [], [], 0
    for t in px.columns:
        for dt in px.index[hit[t]]:
            i = px.index.get_loc(dt)
            before = float(px[t].iloc[i - 1]) if i > 0 else float("nan")
            if not np.isfinite(before) or before < floor or before / tick < min_ticks:
                skipped += 1
                continue
            r1 = float(rets[t].loc[dt])
            implied = 1.0 / (1.0 + r1) if (1.0 + r1) > 0 else float("nan")
            if np.isfinite(implied) and implied >= 1.8 and abs(implied - round(implied)) < 0.08:
                clean.append(f"{t}@{dt.date()}(~{round(implied)}:1)")
            else:
                messy.append(f"{t}@{dt.date()}({implied:.2f})")
    if clean:
        failures.append(f"{label}: {len(clean)} moves imply a clean split ratio, "
                        f"likely unadjusted: {', '.join(clean[:5])}")
    if messy:
        warnings.append(f"{label}: {len(messy)} large drop-and-rebound moves with no "
                        f"clean split ratio (probably real): {', '.join(messy[:3])}")
    if skipped:
        warnings.append(f"{label}: {skipped} large reversals ignored — price below the "
                        f"${floor:.3f} universe screen or within {min_ticks} ticks of the "
                        f"minimum increment, where quantisation fakes clean ratios")

    # ── Flat price runs: still flat at the last bar is what matters ──────
    w = qcfg["stale_weeks"]
    flat = (rets.abs() < 1e-12)
    long_run = (flat.rolling(w).sum() >= w)
    n_any = int(long_run.any().sum())
    n_live = int(long_run.iloc[-1].fillna(False).sum())
    if n_live / n_tick > qcfg["max_dead_ticker_pct"]:
        dead = list(long_run.iloc[-1].fillna(False).pipe(lambda s: s[s]).index[:8])
        failures.append(f"{label}: {n_live} tickers ({n_live / n_tick:.1%}) still flat at "
                        f"the final bar — possible dead feed: {', '.join(dead)}")
    elif n_live:
        warnings.append(f"{label}: {n_live} tickers flat through the final bar "
                        f"(likely suspended or delisted)")
    if n_any - n_live:
        warnings.append(f"{label}: {n_any - n_live} tickers had a >= {w}w flat run that "
                        f"later resumed — normal halts, no effect on the current ranking")

    last_date = px.index[-1]
    age_days = (pd.Timestamp.today().normalize() - last_date.normalize()).days
    if age_days > qcfg["max_staleness_days"]:
        failures.append(f"{label}: latest bar is {age_days}d old ({last_date.date()})")

    return failures, warnings


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
    all_warnings: list[str] = []
    all_warnings: list[str] = []

    for market, label in [("sp500", "S&P 500"), ("asx300", "ASX 300")]:
        u = universe[market]
        payload = fetch_market(u["tickers"], u["names"], u.get("sectors", {}), label, cfg)
        payload["universeUsedFallback"] = u.get("usedFallback", False)
        payload["benchmark"] = fetch_series(cfg["data"]["benchmark"].get(market), start, end)
        payload["riskFree"] = fetch_series(cfg["data"]["risk_free"].get(market), start, end)

        failures, warnings = quality_gate(
            payload, label, cfg["quality"],
            min_price=cfg["universe"]["min_price"].get(market, 0.0))
        payload["qualityProblems"] = failures
        payload["qualityWarnings"] = warnings
        all_problems += failures
        all_warnings += warnings
        atomic_write(DATA_DIR / f"{market}_prices.json", payload)
        print(f"  wrote {market}_prices.json  "
              f"[{len(failures)} failures, {len(warnings)} warnings]")

    if all_warnings:
        print("\nQA warnings (not blocking):")
        for w in all_warnings:
            print(f"  ~ {w}")

    if all_problems:
        # Emergency unblock. QA_SOFT_FAIL=1 downgrades every blocking problem to
        # a loud warning and lets the pipeline continue. It exists so a bad gate
        # calibration cannot hold your whole pipeline hostage on a Saturday —
        # NOT as a way to live with unresolved data problems. Anything published
        # under soft-fail is built on data the gate objected to, so the flag is
        # recorded in the price file and surfaces on the dashboard.
        if os.environ.get("QA_SOFT_FAIL") == "1":
            print("\n" + "!" * 68)
            print("QA_SOFT_FAIL=1 — the following BLOCKING problems were downgraded:")
            for p in all_problems:
                print(f"  ! {p}")
            print("Results are being published from data the quality gate rejected.")
            print("This is a temporary unblock. Fix the cause or add an explicit")
            print("ignore_tickers entry in config.yaml, then remove the flag.")
            print("!" * 68)
            for market in ["sp500", "asx300"]:
                fp = DATA_DIR / f"{market}_prices.json"
                if fp.exists():
                    d = json.loads(fp.read_text())
                    d["qaSoftFailed"] = True
                    atomic_write(fp, d)
            return

        print("\nDATA QUALITY GATE FAILED:")
        for p in all_problems:
            print(f"  - {p}")
        print("\nRun  python scripts/diagnose_data.py <market>  to see the "
              "specific tickers and dates behind these.")
        sys.exit(1)
    print("\nDone. No blocking data quality problems.")


if __name__ == "__main__":
    main()
