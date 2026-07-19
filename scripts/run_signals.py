"""
run_signals.py  (v2)
────────────────────
Computes the live cross-section and writes data/signals_latest.json.

BACKWARD COMPATIBILITY
The existing dashboard reads stocks[].factors with the eight legacy keys
(r121, r91, r61, r31, r2_12, r2_9, r2_6, r2_3) plus price / monthly / chart.
All of those are preserved byte-for-byte in meaning, so the current UI keeps
working untouched. Everything new is additive:

  stocks[].factors   — now also carries the extended factor set
  stocks[].risk      — per-name risk metrics (vol, beta, idio vol, drawdown,
                       Sharpe, Sortino, Ulcer, skew, liquidity)
  stocks[].flags     — data-quality and eligibility flags for that name
  markets[].book     — the constrained portfolio, with the binding constraint
                       recorded per position
  markets[].risk     — portfolio-level risk report
  manifest           — config hash, code version, input fingerprints

The division of labour from v1 is unchanged and deliberate: Python is the
trusted scheduled source of FACTS, the dashboard does the RANKING, so weight
sliders stay live without re-running anything.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as mx        # noqa: E402
import panel as pnl         # noqa: E402
import portfolio as pf      # noqa: E402
from picks_history_writer import upsert_picks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

LEGACY_KEYS = ["r121", "r91", "r61", "r31", "r2_12", "r2_9", "r2_6", "r2_3"]
EXTENDED_KEYS = ["resid_mom", "mom_vol_scaled", "pct_52w_high", "info_discreteness",
                 "st_reversal", "low_vol", "vol_regime", "consistency",
                 "sharpe_12m", "sortino_12m", "ulcer", "drawdown"]
RISK_KEYS = ["vol_12m", "ewma_vol", "beta", "idio_vol"]


def load_config() -> tuple[dict, str]:
    raw = (ROOT / "config.yaml").read_text()
    return yaml.safe_load(raw), hashlib.sha256(raw.encode()).hexdigest()[:12]


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else "missing"


def monthly_history(series: pd.Series, months: int = 12) -> list[dict]:
    s = series.dropna()
    if s.empty:
        return []
    m = s.resample("ME").last().dropna().tail(months + 1)
    out, prev = [], None
    for dt, val in m.items():
        chg = None if prev in (None, 0) else float(val / prev - 1.0)
        out.append({"date": dt.strftime("%Y-%m-%d"),
                    "price": round(float(val), 4),
                    "chg": round(chg, 4) if chg is not None else None})
        prev = val
    return out[-months:]


def run_market(market: str, cfg: dict) -> dict:
    path = DATA_DIR / f"{market}_prices.json"
    with open(path) as f:
        raw = json.load(f)

    dates = pd.to_datetime(raw["dates"])
    prices = pd.DataFrame(raw["prices"], index=dates).sort_index()
    names, sectors_d = raw["names"], raw.get("sectors", {})
    sectors = pd.Series(sectors_d)
    ipo = raw.get("ipoDates", {})
    min_p = cfg["universe"]["min_price"].get(market, 1.0)

    # ── Benchmark: real index if available, equal-weight universe otherwise ──
    rets = prices.pct_change()
    bench_meta = raw.get("benchmark")
    if bench_meta:
        b = pd.Series(bench_meta["values"], index=pd.to_datetime(bench_meta["dates"]))
        mkt_ret = b.reindex(prices.index).pct_change()
        bench_label = bench_meta["symbol"]
    else:
        mkt_ret = rets.mean(axis=1)
        bench_label = "equal-weight universe (no index series in price file)"

    panels = pnl.build_factor_panel(prices, cfg, mkt_ret=mkt_ret)
    pos = len(prices) - 1
    asof = prices.index[pos]

    # ── Liquidity ──────────────────────────────────────────────────────────
    mdv = pd.Series(raw.get("medianDollarVolume60d", {}), dtype=float)
    min_dv = cfg["universe"]["min_median_dollar_volume"].get(market, 0.0)

    stocks, eligible_tickers = [], []
    for t in prices.columns:
        px_now = prices[t].iloc[pos]
        if not np.isfinite(px_now) or px_now < min_p:
            continue

        legacy = {k: panels[k][t].iloc[pos] for k in LEGACY_KEYS if t in panels[k].columns}
        if any(not np.isfinite(v) for v in legacy.values()) or len(legacy) < len(LEGACY_KEYS):
            continue  # matches v1 eligibility: all 8 legacy factors required

        flags = []
        if t in ipo:
            seasoned = asof >= pd.Timestamp(ipo[t]) + pd.Timedelta(
                days=cfg["universe"]["ipo_seasoning_days"])
            if not seasoned:
                flags.append("unseasoned_ipo")
                continue
        dv = float(mdv.get(t, np.nan)) if len(mdv) else np.nan
        if np.isfinite(dv) and dv < min_dv:
            flags.append("below_liquidity_floor")

        ext = {k: panels[k][t].iloc[pos] for k in EXTENDED_KEYS if t in panels[k].columns}
        risk = {k: panels[k][t].iloc[pos] for k in RISK_KEYS if t in panels[k].columns}

        r = rets[t]
        p_arr = prices[t].to_numpy()
        risk.update({
            "max_drawdown_2y": mx.max_drawdown(p_arr),
            "hit_rate_12m": mx.hit_rate(r.tail(52).to_numpy()),
            "tail_ratio": mx.tail_ratio(r.tail(104).to_numpy()),
            "median_dollar_volume_60d": dv if np.isfinite(dv) else None,
        })
        sk, ku = mx.skew_kurt(r.tail(104).to_numpy())
        risk["skew"], risk["excess_kurtosis"] = sk, ku

        def clean(d: dict) -> dict:
            return {k: (round(float(v), 4) if v is not None and np.isfinite(v) else None)
                    for k, v in d.items()}

        stocks.append({
            "ticker": t,
            "name": names.get(t, t),
            "sector": sectors_d.get(t, "—"),
            "price": round(float(px_now), 4),
            "factors": {**clean(legacy), **clean(ext)},
            "risk": clean(risk),
            "flags": flags,
            "monthly": monthly_history(prices[t]),
            "chart": [{"date": d.strftime("%Y-%m-%d"),
                       "price": None if not np.isfinite(v) else round(float(v), 4)}
                      for d, v in prices[t].tail(104).items()],
        })
        if "below_liquidity_floor" not in flags:
            eligible_tickers.append(t)

    # ── Reference book under the legacy equal-weight blend ─────────────────
    score = mx.composite_score(
        pd.DataFrame({k: panels[k].loc[asof] for k in LEGACY_KEYS}).loc[eligible_tickers],
        {k: 1.0 for k in LEGACY_KEYS}, sectors=sectors,
        winsor_sigma=cfg["factors"]["winsor_sigma"],
        sector_neutral=cfg["factors"]["sector_neutral"],
    )
    book = pf.construct(score, sectors, cfg, returns=rets.tail(52),
                        dollar_volume=mdv if len(mdv) else None)
    risk_rep = pf.risk_report(book, rets.tail(52), mkt_ret.tail(52)) if len(book) else {}

    return {
        "market": market,
        "asOf": asof.strftime("%Y-%m-%d"),
        "benchmark": bench_label,
        "universeSize": len(stocks),
        "investableSize": len(eligible_tickers),
        "universeUsedFallback": raw.get("universeUsedFallback", False),
        "qualityProblems": raw.get("qualityProblems", []),
        "stocks": stocks,
        "book": [{"ticker": t, **{k: (None if pd.isna(v) else
                                      (round(float(v), 4) if isinstance(v, (int, float, np.floating))
                                       else v))
                                  for k, v in row.items()}}
                 for t, row in book.iterrows()],
        "bookMeta": {
            "cash_weight": book.attrs.get("cash_weight"),
            "gross_exposure": book.attrs.get("gross_exposure"),
            "ex_ante_vol_annual": book.attrs.get("ex_ante_vol_annual"),
            "max_names_per_sector": book.attrs.get("max_names_per_sector"),
            "displaced_by_sector_cap": book.attrs.get("displaced_by_sector_cap"),
        },
        "risk": risk_rep,
    }


def main() -> None:
    cfg, cfg_hash = load_config()
    out = {
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schemaVersion": cfg["schema_version"],
        "manifest": {
            "configHash": cfg_hash,
            "gitRev": git_rev(),
            "inputs": {f"{m}_prices.json": file_digest(DATA_DIR / f"{m}_prices.json")
                       for m in ["sp500", "asx300"]},
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "markets": {},
    }

    for market in ["sp500", "asx300"]:
        if not (DATA_DIR / f"{market}_prices.json").exists():
            print(f"  skipping {market}: price file not found")
            continue
        print(f"Computing signals for {market}...")
        out["markets"][market] = run_market(market, cfg)
        m = out["markets"][market]
        print(f"  {m['universeSize']} eligible, {m['investableSize']} investable"
              f" at {m['asOf']}")

    path = DATA_DIR / "signals_latest.json"
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"), default=float)
    print(f"Saved {path}")

    # ── Reference snapshot, written IDEMPOTENTLY ──────────────────────────
    # v1 appended unconditionally, which is why picks_history.jsonl accumulated
    # five rows for 2026-06-20. upsert_picks keys on date, so a re-run replaces
    # rather than duplicates and the file is safe to merge.
    snapshot = {"date": out["generatedAt"][:10], "configHash": cfg_hash, "markets": {}}
    for m, md in out["markets"].items():
        snapshot["markets"][m] = [
            {"ticker": r["ticker"], "name": next(
                (s["name"] for s in md["stocks"] if s["ticker"] == r["ticker"]), r["ticker"]),
             "price": next((s["price"] for s in md["stocks"] if s["ticker"] == r["ticker"]), None),
             "weight": r.get("weight"), "sector": r.get("sector")}
            for r in md["book"]
        ]
    upsert_picks(DATA_DIR / "picks_history.jsonl", snapshot)
    print("Upserted snapshot into picks_history.jsonl")


if __name__ == "__main__":
    main()
