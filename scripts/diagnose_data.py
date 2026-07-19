"""
diagnose_data.py
────────────────
Names the tickers behind a QA gate failure so you can judge them, rather than
relaxing a threshold blind.

    python scripts/diagnose_data.py asx300

Distinguishes two very different situations that the gate currently lumps
together:

  * A stock that was SUSPENDED for a few weeks in 2019. Real, common on the
    ASX (capital raisings, takeover talks), and harmless — it affects factor
    values around that date and nothing else.
  * A stock whose feed is dead RIGHT NOW. That corrupts today's ranking, which
    is the only thing you actually trade on.

The original gate treats both as fatal. Only the second one is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RECENT_WEEKS = 26


def load(market: str) -> tuple[pd.DataFrame, dict]:
    with open(DATA_DIR / f"{market}_prices.json") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["prices"], index=pd.to_datetime(raw["dates"])).sort_index()
    return df, raw


def stale_runs(series: pd.Series, min_run: int = 4) -> list[tuple[str, str, int]]:
    """Find maximal runs of identical consecutive prices."""
    s = series.dropna()
    if len(s) < min_run:
        return []
    changed = s.ne(s.shift())
    grp = changed.cumsum()
    out = []
    for _, block in s.groupby(grp):
        if len(block) >= min_run:
            out.append((str(block.index[0].date()), str(block.index[-1].date()), len(block)))
    return out


def main() -> None:
    market = sys.argv[1] if len(sys.argv) > 1 else "asx300"
    df, raw = load(market)
    names = raw.get("names", {})
    last_date = df.index[-1]
    recent_cut = df.index[max(0, len(df) - RECENT_WEEKS)]

    print(f"{market}: {len(df)} weekly bars, {df.shape[1]} tickers, "
          f"{df.index[0].date()} → {last_date.date()}\n")

    # ── Stale price runs ──────────────────────────────────────────────────
    live_problems, historical = [], []
    for t in df.columns:
        runs = stale_runs(df[t])
        if not runs:
            continue
        recent = [r for r in runs if pd.Timestamp(r[1]) >= recent_cut]
        (live_problems if recent else historical).append((t, runs, recent))

    print("=" * 74)
    print(f"STALE PRICE RUNS (4+ identical consecutive weekly closes)")
    print("=" * 74)
    print(f"\nA. LIVE — stale within the last {RECENT_WEEKS} weeks. "
          f"These corrupt TODAY's ranking.\n")
    if live_problems:
        for t, runs, recent in sorted(live_problems, key=lambda x: -max(r[2] for r in x[2])):
            longest = max(recent, key=lambda r: r[2])
            print(f"  {t:<12} {str(names.get(t, ''))[:26]:<28} "
                  f"{longest[2]:>3}w flat  {longest[0]} → {longest[1]}")
    else:
        print("  none")

    print(f"\nB. HISTORICAL — stale only in the past. Almost always a real "
          f"trading halt.\n")
    if historical:
        for t, runs, _ in sorted(historical, key=lambda x: -max(r[2] for r in x[1]))[:20]:
            longest = max(runs, key=lambda r: r[2])
            print(f"  {t:<12} {str(names.get(t, ''))[:26]:<28} "
                  f"{longest[2]:>3}w flat  {longest[0]} → {longest[1]}"
                  f"   ({len(runs)} run{'s' if len(runs) > 1 else ''})")
        if len(historical) > 20:
            print(f"  … and {len(historical) - 20} more")
    else:
        print("  none")

    # ── Suspected unadjusted splits ───────────────────────────────────────
    rets = df.pct_change()
    print("\n" + "=" * 74)
    print("SUSPECTED UNADJUSTED SPLITS (weekly drop >45% immediately reversed >80%)")
    print("=" * 74 + "\n")
    hits = []
    for t in df.columns:
        r = rets[t]
        mask = (r < -0.45) & (r.shift(-1) > 0.8)
        for dt in r.index[mask.fillna(False)]:
            i = df.index.get_loc(dt)
            window = df[t].iloc[max(0, i - 2): i + 3]
            hits.append((t, dt, window))
    if hits:
        for t, dt, window in hits:
            ratio = window.iloc[0] / window.iloc[2] if len(window) > 2 and window.iloc[2] else float("nan")
            print(f"  {t:<12} {str(names.get(t, ''))[:26]:<28} week of {dt.date()}")
            print(f"      prices: {'  '.join(f'{v:.3f}' if np.isfinite(v) else '—' for v in window.values)}")
            print(f"      implied ratio ≈ {ratio:.2f}  "
                  f"{'← looks like a split' if abs(ratio - round(ratio)) < 0.15 and ratio > 1.5 else '← ratio is not a clean split; likely a real move'}\n")
    else:
        print("  none")

    print("=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print("""
  Section A is the one that matters. A stock that has not moved in the last
  six months is either delisted, suspended, or a dead feed — and it is being
  ranked today on prices that are not real. Remove those tickers from your
  base list, or let the liquidity screen catch them.

  Section B is almost always fine. ASX small caps get suspended routinely.
  A halt in 2019 affects factor values around 2019 and nothing else.

  For splits: a clean ratio near 2.00, 3.00 or 10.00 is a genuine unadjusted
  split and the price series is wrong — report it or drop the ticker. A ratio
  like 1.83 is a real crash-and-rebound, which mining and biotech small caps
  do, and the data is fine.
""")


if __name__ == "__main__":
    main()
