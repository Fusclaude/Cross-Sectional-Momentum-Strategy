"""
run_breakout.py
───────────────
All-time-high breakout scanner for the S&P 500 and ASX 300.

Reads the weekly price files fetch_prices.py already produces, so it adds no
extra download. Writes data/breakout_latest.json with, per market:

  candidates   names that broke above their all-time high within the last
               breakout.recent_weeks, have not failed, are not over-extended,
               and clear the universe price and liquidity screens — ordered
               by relative strength
  watchlist    names within 5% of their ATH that have not broken out yet
  eventStudy   what historically happened after breakouts in this universe
  backtest     the rule traded with stops, vs the equal-weight universe
  history      every trade the backtest held in the last history_weeks,
               the strategy's equity curve over that window, and weekly
               price charts for each name, for the dashboard's Breakouts tab

Usage:  python scripts/run_breakout.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breakout as bo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WATCH_WITHIN = 0.05
HISTORY_WEEKS = 104


def _r(v, nd=4):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def run_market(market: str, cfg: dict) -> dict:
    raw = json.loads((DATA_DIR / f"{market}_prices.json").read_text())
    dates = pd.to_datetime(raw["dates"])
    prices = pd.DataFrame(raw["prices"], index=dates).sort_index()
    dv = pd.DataFrame(raw.get("dollarVolume", {}), index=dates).sort_index()
    mdv = pd.Series(raw.get("medianDollarVolume60d", {}), dtype=float)
    bcfg = cfg["breakout"]
    min_p = cfg["universe"]["min_price"].get(market, 0.0)
    min_dv = cfg["universe"]["min_median_dollar_volume"].get(market, 0.0)

    state = bo.breakout_state(prices, bcfg, dollar_volume=dv, min_price=min_p)
    rs = bo.rs_rating(prices, bcfg["rs_weeks"])
    pos = len(prices) - 1
    asof = prices.index[pos]
    history_years = round(len(prices) / bo.PPY, 1)

    def row(t: str) -> dict:
        at = lambda k: state[k][t].iloc[pos]  # noqa: E731
        p = prices[t].iloc[pos]
        level = at("breakout_level")
        return {
            "ticker": t,
            "name": raw["names"].get(t, t),
            "sector": raw.get("sectors", {}).get(t, "—"),
            "price": _r(p),
            "ath": _r(at("ath")),
            "pctFromAth": _r(at("pct_from_ath")),
            "breakoutLevel": _r(level),
            "extension": _r(p / level - 1.0) if np.isfinite(level) else None,
            "weeksSinceBreakout": None if not np.isfinite(at("weeks_since_breakout"))
            else int(at("weeks_since_breakout")),
            "breakoutDate": None if not np.isfinite(at("weeks_since_breakout"))
            else prices.index[pos - int(at("weeks_since_breakout"))].strftime("%Y-%m-%d"),
            "baseWeeks": None if not np.isfinite(at("base_weeks")) else int(at("base_weeks")),
            "breakoutVolumeRatio": _r(at("breakout_volume_ratio"), 2),
            "stopPrice": _r(level * (1.0 - bcfg["stop_pct"])) if np.isfinite(level) else None,
            "rsRating": None if not np.isfinite(rs[t].iloc[pos]) else int(rs[t].iloc[pos]),
            "medianDollarVolume60d": _r(mdv.get(t, np.nan), 0),
        }

    liquid = lambda t: not (np.isfinite(mdv.get(t, np.nan)) and mdv[t] < min_dv)  # noqa: E731
    cand_mask = state["candidate"].iloc[pos].fillna(0.0).astype(bool)
    candidates = [row(t) for t in cand_mask[cand_mask].index if liquid(t)]
    candidates.sort(key=lambda r: -(r["rsRating"] or 0))

    near = state["pct_from_ath"].iloc[pos]
    watch = [row(t) for t in near.index
             if np.isfinite(near[t]) and -WATCH_WITHIN <= near[t] < 0 and not cand_mask[t]
             and prices[t].iloc[pos] >= min_p and liquid(t)
             and prices[t].iloc[:pos + 1].notna().sum() > bcfg["min_history_weeks"]]
    watch.sort(key=lambda r: -(r["pctFromAth"] or -1))

    cost = cfg["costs"]["spread_bps"].get(market, 10.0) + cfg["costs"]["commission_bps"]
    bt = bo.backtest(state, prices, bcfg, rs, cost_bps=cost)
    history = recent_history(bt, prices, raw, [c["ticker"] for c in candidates + watch[:40]])
    bt.pop("tradeLog", None)
    bt.pop("series", None)
    return {
        "market": market,
        "asOf": asof.strftime("%Y-%m-%d"),
        "historyYears": history_years,
        "universeSize": int(prices.iloc[pos].notna().sum()),
        "atNewHighThisWeek": int((state["pct_from_ath"].iloc[pos] == 0).sum()),
        "candidates": candidates,
        "watchlist": watch[:40],
        "eventStudy": bo.event_study(state, prices, bcfg["event_horizons_weeks"]),
        "backtest": bt,
        "history": history,
    }


def recent_history(bt: dict, prices: pd.DataFrame, raw: dict, extra: list[str]) -> dict:
    """
    The last HISTORY_WEEKS of the traded rule: every position held at any
    point in the window (including ones bought before it), the equity curve
    against the equal-weight universe, and a weekly price chart for each name
    traded or currently listed, so the dashboard can mark buys and sells.
    """
    start = prices.index[max(0, len(prices) - HISTORY_WEEKS)]
    if "error" in bt:
        return {"windowStart": start.strftime("%Y-%m-%d"), "trades": [], "charts": {}}
    ds = lambda d: None if d is None else d.strftime("%Y-%m-%d")  # noqa: E731
    trades = [{
        "ticker": t["ticker"],
        "name": raw["names"].get(t["ticker"], t["ticker"]),
        "sector": raw.get("sectors", {}).get(t["ticker"], "—"),
        "entryDate": ds(t["entry_date"]), "entryPrice": _r(t["entry_px"]),
        "pivot": _r(t["pivot"]),
        "exitDate": ds(t["exit_date"]), "exitPrice": _r(t["exit_px"]),
        "return": _r(t["return"]), "reason": t["reason"],
        "weeksHeld": int(((t["exit_date"] or prices.index[-1]) - t["entry_date"]).days // 7),
    } for t in bt["tradeLog"] if t["exit_date"] is None or t["exit_date"] >= start]
    trades.sort(key=lambda t: t["entryDate"], reverse=True)

    closed = [t["return"] for t in trades if t["exitDate"] and t["return"] is not None]
    ser = bt["series"].loc[bt["series"].index > start]
    curve = (1.0 + ser[["strategy", "universe"]]).cumprod()
    tail = prices.iloc[-HISTORY_WEEKS:]
    names = sorted({t["ticker"] for t in trades} | set(extra))
    return {
        "windowStart": start.strftime("%Y-%m-%d"),
        "weeks": len(ser),
        "summary": {
            "trades": len(trades),
            "open": sum(1 for t in trades if t["exitDate"] is None),
            "closed": len(closed),
            "winRate": _r(float(np.mean([r > 0 for r in closed]))) if closed else None,
            "avgReturn": _r(float(np.mean(closed))) if closed else None,
            "medianReturn": _r(float(np.median(closed))) if closed else None,
            "strategy": {k: _r(v) for k, v in bo.perf(ser["strategy"].to_numpy()).items()}
            if len(ser) > 2 else None,
            "universe": {k: _r(v) for k, v in bo.perf(ser["universe"].to_numpy()).items()}
            if len(ser) > 2 else None,
            "avgPositions": _r(float(ser["positions"].mean()), 1) if len(ser) else None,
        },
        "equity": {"dates": [d.strftime("%Y-%m-%d") for d in curve.index],
                   "strategy": [_r(v) for v in curve["strategy"]],
                   "universe": [_r(v) for v in curve["universe"]]},
        "trades": trades,
        "chartDates": [d.strftime("%Y-%m-%d") for d in tail.index],
        "charts": {t: [_r(v) for v in tail[t]] for t in names if t in tail.columns},
    }


def _pct(v):
    return "   n/a" if v is None else f"{v * 100:6.1f}%"


def print_market(m: dict) -> None:
    print(f"\n══ {m['market'].upper()}  as of {m['asOf']}  "
          f"({m['historyYears']}y history, {m['atNewHighThisWeek']} closed at a new high this week)")
    print(f"  {len(m['candidates'])} breakout candidates:")
    if m["candidates"]:
        print(f"    {'ticker':<9}{'RS':>4}{'broke out':>12}{'base':>6}{'pivot':>10}"
              f"{'price':>10}{'ext':>8}{'stop':>10}{'vol x':>7}")
    for c in m["candidates"]:
        vr = c["breakoutVolumeRatio"]
        print(f"    {c['ticker']:<9}{c['rsRating'] or 0:>4}{c['breakoutDate']:>12}"
              f"{c['baseWeeks']:>5}w{c['breakoutLevel']:>10.2f}{c['price']:>10.2f}"
              f"{_pct(c['extension']):>8}{c['stopPrice']:>10.2f}"
              f"{('' if vr is None else f'{vr:.1f}'):>7}")
    es = m["eventStudy"]
    print(f"  Event study, {es['n_events']} historical breakouts "
          f"(forward return / excess vs equal-weight universe):")
    for h, s in es["horizons"].items():
        print(f"    {h:>3}w  mean {_pct(s['mean'])}  median {_pct(s['median'])}  "
              f"excess {_pct(s['mean_excess'])}  beat universe {s['excess_hit_rate']:.0%}")
    bt = m["backtest"]
    if "error" not in bt:
        s, b = bt["strategy"], bt["equal_weight_universe"]
        print(f"  Backtest ({s['years']}y, avg {bt['avg_positions']:.1f} positions): "
              f"CAGR {_pct(s['cagr'])} vs {_pct(b['cagr'])}, "
              f"Sharpe {s['sharpe'] or 0:.2f} vs {b['sharpe'] or 0:.2f}, "
              f"maxDD {_pct(s['max_drawdown'])} vs {_pct(b['max_drawdown'])}")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    out = {"generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
           "params": cfg["breakout"], "markets": {}}
    for market in ["sp500", "asx300"]:
        if not (DATA_DIR / f"{market}_prices.json").exists():
            print(f"  skipping {market}: price file not found")
            continue
        out["markets"][market] = run_market(market, cfg)
        print_market(out["markets"][market])

    path = DATA_DIR / "breakout_latest.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
