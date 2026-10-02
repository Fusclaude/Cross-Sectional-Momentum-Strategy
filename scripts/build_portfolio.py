"""
build_portfolio.py
──────────────────
Turns data/trades.csv (your actual fills) into data/portfolio.json, which the
dashboard's Portfolio tab renders:

  open      positions still held, valued at the latest weekly close
  closed    every sale, matched to the shares it sold first-in-first-out,
            with the realised profit or loss
  weekly    one row per week from portfolio.history_start: market value,
            cost of open positions, unrealised and realised P/L, cash put in

Prices come from the weekly price files fetch_prices.py already writes, so
nothing extra is downloaded for names in the S&P 500 or ASX 300. Anything you
traded outside those indexes (ETFs, small caps) is fetched here; if that
fetch fails, those weeks are valued at your own last trade price and flagged
as estimates rather than silently dropped.

All amounts are cash amounts INCLUDING brokerage, so realised P/L is net of
costs. Prices are dividend-adjusted, as elsewhere in the pipeline: the latest
close is the real price, earlier weeks of dividend payers read slightly low.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_trades(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    trades = []
    for row in csv.DictReader(lines):
        side = row["side"].strip().upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"bad side {row['side']!r} in {row}")
        trades.append({"date": pd.Timestamp(row["date"].strip()),
                       "ticker": row["ticker"].strip().upper(), "side": side,
                       "shares": int(row["shares"]), "amount": float(row["amount"])})
    # Stable sort: same-day trades keep file order, so a buy listed before a
    # sell of the same name on the same day is matched correctly.
    return sorted(trades, key=lambda t: t["date"])


def match_fifo(trades: list[dict]) -> tuple[list[dict], dict[str, deque]]:
    """Replay trades in date order. Returns (closed sales, open lots by ticker)."""
    lots: dict[str, deque] = defaultdict(deque)   # ticker -> [date, shares, cost/share]
    closed = []
    for t in trades:
        if t["side"] == "BUY":
            lots[t["ticker"]].append([t["date"], t["shares"], t["amount"] / t["shares"]])
            continue
        rem, cost, first = t["shares"], 0.0, None
        if rem > sum(lot[1] for lot in lots[t["ticker"]]):
            raise ValueError(f"{t['date'].date()} sells {rem} {t['ticker']} but fewer are held")
        while rem:
            lot = lots[t["ticker"]][0]
            n = min(rem, lot[1])
            cost += n * lot[2]
            first = first or lot[0]
            lot[1] -= n
            rem -= n
            if not lot[1]:
                lots[t["ticker"]].popleft()
        closed.append({"ticker": t["ticker"], "firstBuy": first, "sold": t["date"],
                       "shares": t["shares"], "cost": cost, "proceeds": t["amount"],
                       "pnl": t["amount"] - cost, "pnlPct": (t["amount"] - cost) / cost})
    return closed, lots


def load_prices(tickers: set[str], start: pd.Timestamp) -> tuple[pd.DataFrame, dict, set[str]]:
    """Weekly closes for every ticker traded, keyed by ASX code. Returns
    (prices, names, tickers that still have no price data)."""
    frames, names = [], {}
    for market, suffix in [("asx300", ".AX"), ("sp500", "")]:
        path = DATA_DIR / f"{market}_prices.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        idx = pd.to_datetime(raw["dates"])
        have = {t: t + suffix for t in tickers if (t + suffix) in raw["prices"]}
        if have:
            frames.append(pd.DataFrame({t: raw["prices"][k] for t, k in have.items()}, index=idx))
            names.update({t: raw["names"].get(k, t) for t, k in have.items()})
    prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    prices = prices.loc[:, ~prices.columns.duplicated()]

    missing = tickers - set(prices.columns)
    if missing:
        fetched = fetch_missing(missing, start)
        if not fetched.empty:
            prices = prices.join(fetched, how="outer") if not prices.empty else fetched
    missing = tickers - set(prices.columns)
    return prices.sort_index(), names, missing


def fetch_missing(tickers: set[str], start: pd.Timestamp) -> pd.DataFrame:
    """Names outside both indexes. Same settings as fetch_prices.py."""
    try:
        import yfinance as yf
        df = yf.download([t + ".AX" for t in sorted(tickers)], start=start - timedelta(days=14),
                         interval="1d", auto_adjust=True, progress=False, group_by="ticker")
    except Exception as e:  # network or API failure: fall back to trade prices
        print(f"  WARNING: price fetch failed for {sorted(tickers)} ({e})")
        return pd.DataFrame()
    out = {}
    for t in tickers:
        try:
            col = df[(t + ".AX", "Close")] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
        except KeyError:
            continue
        wk = col.dropna().resample("W-FRI").last().dropna()
        if len(wk):
            out[t] = wk
    if tickers - set(out):
        print(f"  WARNING: no price data for {sorted(tickers - set(out))}")
    return pd.DataFrame(out)


def weekly_history(trades: list[dict], prices: pd.DataFrame, start: pd.Timestamp) -> list[dict]:
    """
    One row per Friday from `start` to the latest price. Holdings at a week are
    every trade dated on or before that Friday. A name with no close that week
    is valued at its most recent close; a name with no price data at all is
    valued at your last trade price in it and listed under `estimated`.
    """
    weeks = prices.index[prices.index >= start] if len(prices) else pd.DatetimeIndex([])
    filled = prices.ffill()
    rows = []
    for wk in weeks:
        upto = [t for t in trades if t["date"] <= wk]
        closed, lots = match_fifo(upto)
        last_trade_px = {t["ticker"]: t["amount"] / t["shares"] for t in upto}
        positions, value, cost, estimated = [], 0.0, 0.0, []
        for tkr, ls in sorted(lots.items()):
            sh = sum(lot[1] for lot in ls)
            if not sh:
                continue
            c = sum(lot[1] * lot[2] for lot in ls)
            px = filled.at[wk, tkr] if tkr in filled.columns else float("nan")
            if not pd.notna(px):
                px = last_trade_px[tkr]
                estimated.append(tkr)
            positions.append({"ticker": tkr, "shares": sh, "price": round(float(px), 4),
                              "value": round(sh * float(px), 2), "cost": round(c, 2)})
            value += sh * float(px)
            cost += c
        realised = sum(c["pnl"] for c in closed)
        invested = sum(t["amount"] * (1 if t["side"] == "BUY" else -1) for t in upto)
        rows.append({
            "week": wk.strftime("%Y-%m-%d"),
            "marketValue": round(value, 2),
            "openCost": round(cost, 2),
            "unrealised": round(value - cost, 2),
            "realisedToDate": round(realised, 2),
            "totalPnl": round(value - cost + realised, 2),
            "netInvested": round(invested, 2),
            "positions": positions,
            "estimated": estimated,
        })
    for prev, row in zip([None] + rows[:-1], rows):
        row["weekChange"] = None if prev is None else round(row["totalPnl"] - prev["totalPnl"], 2)
    return rows


def build(trades: list[dict], prices: pd.DataFrame, names: dict, missing: set[str],
          start: pd.Timestamp) -> dict:
    closed, lots = match_fifo(trades)
    latest = prices.ffill().iloc[-1] if len(prices) else pd.Series(dtype=float)
    last_trade_px = {t["ticker"]: t["amount"] / t["shares"] for t in trades}
    ds = lambda d: d.strftime("%Y-%m-%d")  # noqa: E731

    open_pos = []
    for tkr, ls in sorted(lots.items()):
        sh = sum(lot[1] for lot in ls)
        if not sh:
            continue
        cost = sum(lot[1] * lot[2] for lot in ls)
        px = latest.get(tkr, float("nan"))
        est = not pd.notna(px)
        px = float(last_trade_px[tkr] if est else px)
        open_pos.append({
            "ticker": tkr, "name": names.get(tkr, tkr), "shares": sh,
            "cost": round(cost, 2), "avgPrice": round(cost / sh, 4),
            "firstBuy": ds(ls[0][0]), "buys": len(ls),
            "price": round(px, 4), "marketValue": round(sh * px, 2),
            "unrealised": round(sh * px - cost, 2),
            "unrealisedPct": round((sh * px - cost) / cost, 4),
            "estimated": est,
        })
    open_pos.sort(key=lambda p: -p["marketValue"])

    sold = [{**c, "name": names.get(c["ticker"], c["ticker"]),
             "firstBuy": ds(c["firstBuy"]), "sold": ds(c["sold"]),
             "cost": round(c["cost"], 2), "proceeds": round(c["proceeds"], 2),
             "pnl": round(c["pnl"], 2), "pnlPct": round(c["pnlPct"], 4)}
            for c in sorted(closed, key=lambda c: c["sold"], reverse=True)]

    mv = sum(p["marketValue"] for p in open_pos)
    oc = sum(p["cost"] for p in open_pos)
    realised = sum(c["pnl"] for c in closed)
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "priceDate": ds(prices.index[-1]) if len(prices) else None,
        "historyStart": ds(start),
        "noPriceData": sorted(missing),
        "summary": {
            "marketValue": round(mv, 2), "openCost": round(oc, 2),
            "unrealised": round(mv - oc, 2),
            "unrealisedPct": round((mv - oc) / oc, 4) if oc else None,
            "realised": round(realised, 2),
            "realisedPct": round(realised / sum(c["cost"] for c in closed), 4) if closed else None,
            "totalPnl": round(mv - oc + realised, 2),
            "closedTrades": len(closed),
            "winners": sum(1 for c in closed if c["pnl"] > 0),
            "openPositions": len(open_pos),
        },
        "open": open_pos,
        "closed": sold,
        "weekly": weekly_history(trades, prices, start),
    }


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    pcfg = cfg.get("my_trades", {})
    trades = load_trades(ROOT / pcfg.get("trades_file", "data/trades.csv"))
    start = pd.Timestamp(pcfg.get("history_start", "2026-06-22"))
    prices, names, missing = load_prices({t["ticker"] for t in trades},
                                         min(t["date"] for t in trades))
    out = build(trades, prices, names, missing, start)
    (DATA_DIR / "portfolio.json").write_text(json.dumps(out, separators=(",", ":")))

    s = out["summary"]
    print(f"Portfolio as of {out['priceDate']}: {s['openPositions']} open, "
          f"value ${s['marketValue']:,.2f} vs cost ${s['openCost']:,.2f} "
          f"(unrealised {s['unrealised']:+,.2f}); realised {s['realised']:+,.2f} "
          f"over {s['closedTrades']} sales; {len(out['weekly'])} weekly rows")
    if missing:
        print(f"  no price data for {', '.join(sorted(missing))}: valued at trade prices")
    print("Saved data/portfolio.json")


if __name__ == "__main__":
    sys.exit(main())
