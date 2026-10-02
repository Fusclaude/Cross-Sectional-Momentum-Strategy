"""
build_portfolio.py
──────────────────
Turns your record of trades into data/portfolio.json, which the dashboard's
Portfolio tab renders (settings under my_trades: in config.yaml):

  data/trades.csv   buys, sells, dividends and free share grants, ASX and US
  data/cash.csv     deposits and AUD→USD conversions

Output:
  open      positions still held, valued at the latest weekly close
  closed    every sale, matched to the shares it sold first-in-first-out
  weekly    one row per week from history_start: market value, cost of open
            positions, unrealised and realised P/L, all in AUD
  cash      deposits, conversions, and USD not yet invested

CURRENCY
Each position is reported in its own currency. Totals are in AUD:
  * a US buy costs what you actually paid for the USD — its USD amount at the
    rate of your most recent conversion on or before the trade;
  * US values and sale proceeds use the market AUD/USD rate for that week
    (fetched in the workflow; if that fails, your latest conversion rate).
So AUD P/L on a US stock includes the currency move, as it does in reality.

PRICES come from the weekly price files fetch_prices.py already writes, so
nothing extra is downloaded for names in the S&P 500 or ASX 300. Anything
else is fetched here; if that fails its weeks are valued at your last trade
price and flagged as estimates. Prices are dividend-adjusted, as elsewhere in
the pipeline: the latest close is the real price, older weeks of dividend
payers read slightly low. Amounts include brokerage, so P/L is net of costs.
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
EPS = 1e-9   # fractional shares: treat anything this small as zero

# Where each market's names live in the price files, and their Yahoo suffix.
MARKETS = {"ASX": ("asx300", ".AX", "AUD"), "US": ("sp500", "", "USD")}


def _rows(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def key(code: str, market: str) -> str:
    """How a holding is named everywhere: CDA for ASX, IREN.US for US."""
    return code if market == "ASX" else f"{code}.{market}"


def load_trades(path: Path) -> list[dict]:
    trades = []
    for row in _rows(path):
        side = row["side"].strip().upper()
        market = (row.get("market") or "ASX").strip().upper()
        if side not in ("BUY", "SELL", "DIVIDEND", "GRANT"):
            raise ValueError(f"bad side {row['side']!r} in {row}")
        if market not in MARKETS:
            raise ValueError(f"bad market {market!r} in {row}")
        code = row["ticker"].strip().upper()
        trades.append({
            "date": pd.Timestamp(row["date"].strip()),
            "code": code, "market": market, "ticker": key(code, market),
            "side": side,
            "shares": float(row["shares"]) if (row.get("shares") or "").strip() else 0.0,
            "amount": float(row["amount"]),
            "ccy": (row.get("currency") or MARKETS[market][2]).strip().upper(),
        })
    # Stable sort: same-day trades keep file order.
    return sorted(trades, key=lambda t: t["date"])


def load_cash(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for row in _rows(path):
        f = lambda k: float(row[k]) if (row.get(k) or "").strip() else 0.0  # noqa: E731
        out.append({"date": pd.Timestamp(row["date"].strip()),
                    "type": row["type"].strip().upper(), "aud": f("aud"), "usd": f("usd")})
    return sorted(out, key=lambda c: c["date"])


class FX:
    """
    USD per AUD. `paid(date)` is the rate of your latest conversion on or before
    `date` (what your USD actually cost); `market(date)` is the weekly market
    rate, falling back to `paid` when no market series is available.
    """
    def __init__(self, cash: list[dict], market: pd.Series | None = None):
        conv = [(c["date"], c["usd"] / c["aud"]) for c in cash if c["type"] == "FX" and c["aud"] > 0]
        self.conversions = pd.Series(dict(conv)).sort_index() if conv else pd.Series(dtype=float)
        self.conversions = self.conversions.groupby(level=0).last()
        self.market_series = market.dropna().sort_index() if market is not None and len(market) else None

    @staticmethod
    def _asof(s: pd.Series, date) -> float | None:
        if s is None or s.empty:
            return None
        before = s.loc[:date]
        return float(before.iloc[-1] if len(before) else s.iloc[0])

    def paid(self, date) -> float:
        r = self._asof(self.conversions, date)
        return r if r else self.market(date)

    def market(self, date) -> float:
        r = self._asof(self.market_series, date)
        if r:
            return r
        r = self._asof(self.conversions, date)
        if r:
            return r
        raise ValueError("US trades need an FX rate: add a conversion to data/cash.csv")

    def to_aud(self, amount: float, ccy: str, rate: float) -> float:
        return amount if ccy == "AUD" else amount / rate


def match_fifo(trades: list[dict], fx: FX | None = None) -> tuple[list[dict], dict[str, deque], list[dict]]:
    """
    Replay trades in date order. Returns (closed sales, open lots by ticker,
    dividends). Each lot is [date, shares, cost/share, AUD cost/share].
    """
    lots: dict[str, deque] = defaultdict(deque)
    closed, dividends = [], []
    for t in trades:
        ccy = t["ccy"]
        if t["side"] == "DIVIDEND":
            aud = fx.to_aud(t["amount"], ccy, fx.market(t["date"])) if ccy != "AUD" else t["amount"]
            dividends.append({**t, "aud": aud})
            continue
        if t["side"] in ("BUY", "GRANT"):
            cost = 0.0 if t["side"] == "GRANT" else t["amount"]
            aud = cost if ccy == "AUD" else fx.to_aud(cost, ccy, fx.paid(t["date"]))
            lots[t["ticker"]].append([t["date"], t["shares"], cost / t["shares"], aud / t["shares"]])
            continue
        rem, cost, cost_aud, first = t["shares"], 0.0, 0.0, None
        if rem > sum(lot[1] for lot in lots[t["ticker"]]) + EPS:
            raise ValueError(f"{t['date'].date()} sells {rem} {t['ticker']} but fewer are held")
        while rem > EPS:
            lot = lots[t["ticker"]][0]
            n = min(rem, lot[1])
            cost += n * lot[2]
            cost_aud += n * lot[3]
            first = first or lot[0]
            lot[1] -= n
            rem -= n
            if lot[1] <= EPS:
                lots[t["ticker"]].popleft()
        proceeds_aud = t["amount"] if ccy == "AUD" else fx.to_aud(t["amount"], ccy, fx.market(t["date"]))
        closed.append({"ticker": t["ticker"], "market": t["market"], "currency": ccy,
                       "firstBuy": first, "sold": t["date"], "shares": t["shares"],
                       "cost": cost, "proceeds": t["amount"], "pnl": t["amount"] - cost,
                       "pnlPct": (t["amount"] - cost) / cost if cost else None,
                       "costAud": cost_aud, "proceedsAud": proceeds_aud,
                       "pnlAud": proceeds_aud - cost_aud})
    return closed, lots, dividends


def load_prices(trades: list[dict], start: pd.Timestamp) -> tuple[pd.DataFrame, dict, set[str]]:
    """Weekly closes keyed by holding name. Returns (prices, names, still missing)."""
    wanted = {t["ticker"]: (t["code"], t["market"]) for t in trades if t["side"] != "DIVIDEND"}
    frames, names = [], {}
    for market, (file, suffix, _) in MARKETS.items():
        path = DATA_DIR / f"{file}_prices.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        have = {k: code + suffix for k, (code, m) in wanted.items()
                if m == market and (code + suffix) in raw["prices"]}
        if have:
            frames.append(pd.DataFrame({k: raw["prices"][y] for k, y in have.items()},
                                       index=pd.to_datetime(raw["dates"])))
            names.update({k: raw["names"].get(y, k) for k, y in have.items()})
    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

    missing = {k: code + MARKETS[m][1] for k, (code, m) in wanted.items() if k not in prices.columns}
    if missing:
        fetched = _download(list(missing.values()), start)
        got = {k: fetched[y] for k, y in missing.items() if y in fetched}
        if got:
            extra = pd.DataFrame(got)
            prices = prices.join(extra, how="outer") if not prices.empty else extra
    return prices.sort_index(), names, set(wanted) - set(prices.columns)


def _download(symbols: list[str], start: pd.Timestamp) -> dict[str, pd.Series]:
    """Weekly (W-FRI) closes from Yahoo, same settings as fetch_prices.py."""
    try:
        import yfinance as yf
        df = yf.download(symbols, start=start - timedelta(days=14), interval="1d",
                         auto_adjust=True, progress=False, group_by="ticker")
    except Exception as e:  # network or API failure: callers fall back
        print(f"  WARNING: price fetch failed for {symbols} ({e})")
        return {}
    out = {}
    for y in symbols:
        try:
            col = df[(y, "Close")] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
        except KeyError:
            continue
        wk = col.dropna().resample("W-FRI").last().dropna()
        if len(wk):
            out[y] = wk
    if set(symbols) - set(out):
        print(f"  WARNING: no price data for {sorted(set(symbols) - set(out))}")
    return out


def _value(lots, prices_row, last_trade_px, fx_rate, ccy_of) -> tuple[list[dict], list[str]]:
    """Value every open lot. Returns (positions, tickers valued at trade price)."""
    positions, estimated = [], []
    for tkr, ls in sorted(lots.items()):
        sh = sum(lot[1] for lot in ls)
        if sh <= EPS:
            continue
        cost = sum(lot[1] * lot[2] for lot in ls)
        cost_aud = sum(lot[1] * lot[3] for lot in ls)
        px = prices_row.get(tkr, float("nan")) if prices_row is not None else float("nan")
        est = not pd.notna(px)
        px = float(last_trade_px.get(tkr, 0.0) if est else px)
        if est:
            estimated.append(tkr)
        ccy = ccy_of[tkr]
        value = sh * px
        value_aud = value if ccy == "AUD" else value / fx_rate
        positions.append({"ticker": tkr, "currency": ccy, "shares": sh, "price": px,
                          "cost": cost, "value": value, "costAud": cost_aud,
                          "valueAud": value_aud, "firstBuy": ls[0][0], "lots": len(ls),
                          "estimated": est})
    return positions, estimated


def _last_trade_prices(trades: list[dict]) -> dict[str, float]:
    return {t["ticker"]: t["amount"] / t["shares"] for t in trades
            if t["side"] in ("BUY", "SELL") and t["shares"] > 0}


def weekly_history(trades: list[dict], prices: pd.DataFrame, start: pd.Timestamp,
                   fx: FX | None = None) -> list[dict]:
    """
    One row per Friday from `start` to the latest price, in AUD. Holdings at a
    week are every trade dated on or before that Friday; a name with no close
    that week uses its latest close, and one with no price data at all its
    last trade price (listed under `estimated`).
    """
    fx = fx or FX([])
    ccy_of = {t["ticker"]: t["ccy"] for t in trades}
    weeks = prices.index[prices.index >= start] if len(prices) else pd.DatetimeIndex([])
    filled = prices.ffill()
    rows = []
    for wk in weeks:
        upto = [t for t in trades if t["date"] <= wk]
        closed, lots, divs = match_fifo(upto, fx)
        has_usd = any(c != "AUD" for c in ccy_of.values())
        rate = fx.market(wk) if has_usd else 1.0
        pos, est = _value(lots, filled.loc[wk], _last_trade_prices(upto), rate, ccy_of)
        value = sum(p["valueAud"] for p in pos)
        cost = sum(p["costAud"] for p in pos)
        realised = sum(c["pnlAud"] for c in closed) + sum(d["aud"] for d in divs)
        bought = sum(fx.to_aud(t["amount"], t["ccy"], fx.paid(t["date"]) if t["ccy"] != "AUD" else 1.0)
                     for t in upto if t["side"] == "BUY")
        invested = bought - sum(c["proceedsAud"] for c in closed)
        rows.append({
            "week": wk.strftime("%Y-%m-%d"),
            "marketValue": round(value, 2),
            "openCost": round(cost, 2),
            "unrealised": round(value - cost, 2),
            "realisedToDate": round(realised, 2),
            "totalPnl": round(value - cost + realised, 2),
            "netInvested": round(invested, 2),
            "audUsd": round(rate, 4) if has_usd else None,
            "positions": [{"ticker": p["ticker"], "currency": p["currency"],
                           "shares": round(p["shares"], 5), "price": round(p["price"], 4),
                           "value": round(p["value"], 2), "valueAud": round(p["valueAud"], 2),
                           "cost": round(p["costAud"], 2)} for p in pos],
            "estimated": est,
        })
    for prev, row in zip([None] + rows[:-1], rows):
        row["weekChange"] = None if prev is None else round(row["totalPnl"] - prev["totalPnl"], 2)
    return rows


def cash_summary(cash: list[dict], trades: list[dict]) -> dict:
    """Deposits, conversions and the USD left over after US trades."""
    us = [t for t in trades if t["ccy"] == "USD"]
    usd_in = sum(c["usd"] for c in cash if c["type"] == "FX")
    usd_out = sum(t["amount"] for t in us if t["side"] == "BUY")
    usd_back = sum(t["amount"] for t in us if t["side"] in ("SELL", "DIVIDEND"))
    aud_conv = sum(c["aud"] for c in cash if c["type"] == "FX")
    return {
        "depositedAud": round(sum(c["aud"] for c in cash if c["type"] == "DEPOSIT"), 2),
        "convertedAud": round(aud_conv, 2),
        "convertedUsd": round(usd_in, 2),
        "avgRate": round(usd_in / aud_conv, 4) if aud_conv else None,
        "usdCash": round(usd_in - usd_out + usd_back, 2) if cash else None,
        "firstDeposit": min((c["date"] for c in cash), default=None),
    }


def build(trades: list[dict], prices: pd.DataFrame, names: dict, missing: set[str],
          start: pd.Timestamp, fx: FX | None = None, cash: list[dict] | None = None) -> dict:
    fx = fx or FX(cash or [])
    ccy_of = {t["ticker"]: t["ccy"] for t in trades}
    has_usd = any(c != "AUD" for c in ccy_of.values())
    closed, lots, divs = match_fifo(trades, fx)
    asof = prices.index[-1] if len(prices) else pd.Timestamp.today().normalize()
    rate = fx.market(asof) if has_usd else 1.0
    latest = prices.ffill().iloc[-1] if len(prices) else None
    pos, _ = _value(lots, latest, _last_trade_prices(trades), rate, ccy_of)
    ds = lambda d: d.strftime("%Y-%m-%d")  # noqa: E731
    r2 = lambda v: None if v is None else round(v, 2)  # noqa: E731

    open_pos = [{
        "ticker": p["ticker"], "name": names.get(p["ticker"], p["ticker"]),
        "currency": p["currency"], "shares": round(p["shares"], 5),
        "cost": r2(p["cost"]), "avgPrice": round(p["cost"] / p["shares"], 4),
        "firstBuy": ds(p["firstBuy"]), "buys": p["lots"],
        "price": round(p["price"], 4), "marketValue": r2(p["value"]),
        "unrealised": r2(p["value"] - p["cost"]),
        "unrealisedPct": round((p["value"] - p["cost"]) / p["cost"], 4) if p["cost"] else None,
        "costAud": r2(p["costAud"]), "valueAud": r2(p["valueAud"]),
        "unrealisedAud": r2(p["valueAud"] - p["costAud"]),
        "estimated": p["estimated"],
    } for p in pos]
    open_pos.sort(key=lambda p: -p["valueAud"])

    sold = [{"ticker": c["ticker"], "name": names.get(c["ticker"], c["ticker"]),
             "currency": c["currency"], "firstBuy": ds(c["firstBuy"]), "sold": ds(c["sold"]),
             "shares": round(c["shares"], 5), "cost": r2(c["cost"]), "proceeds": r2(c["proceeds"]),
             "pnl": r2(c["pnl"]), "pnlPct": None if c["pnlPct"] is None else round(c["pnlPct"], 4),
             "costAud": r2(c["costAud"]), "proceedsAud": r2(c["proceedsAud"]),
             "pnlAud": r2(c["pnlAud"])}
            for c in sorted(closed, key=lambda c: c["sold"], reverse=True)]

    mv = sum(p["valueAud"] for p in pos)
    oc = sum(p["costAud"] for p in pos)
    div_aud = sum(d["aud"] for d in divs)
    realised = sum(c["pnlAud"] for c in closed) + div_aud
    sold_cost = sum(c["costAud"] for c in closed)
    cs = cash_summary(cash or [], trades)
    if cs["firstDeposit"] is not None:
        cs["firstDeposit"] = ds(cs["firstDeposit"])
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "priceDate": ds(asof),
        "historyStart": ds(start),
        "noPriceData": sorted(missing),
        "fx": {"audUsd": round(rate, 4) if has_usd else None,
               "source": "market" if fx.market_series is not None else "your latest conversion"},
        "summary": {
            "marketValue": r2(mv), "openCost": r2(oc), "unrealised": r2(mv - oc),
            "unrealisedPct": round((mv - oc) / oc, 4) if oc else None,
            "realised": r2(realised),
            "realisedPct": round(realised / sold_cost, 4) if sold_cost else None,
            "dividends": r2(div_aud),
            "totalPnl": r2(mv - oc + realised),
            "closedTrades": len(closed),
            "winners": sum(1 for c in closed if c["pnlAud"] > 0),
            "openPositions": len(open_pos),
        },
        "cash": cs,
        "dividends": [{"ticker": d["ticker"], "date": ds(d["date"]), "amount": d["amount"],
                       "currency": d["ccy"], "aud": r2(d["aud"])} for d in divs],
        "open": open_pos,
        "closed": sold,
        "weekly": weekly_history(trades, prices, start, fx),
    }


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    pcfg = cfg.get("my_trades", {})
    trades = load_trades(ROOT / pcfg.get("trades_file", "data/trades.csv"))
    cash = load_cash(ROOT / pcfg.get("cash_file", "data/cash.csv"))
    start = pd.Timestamp(pcfg.get("history_start", "2026-06-22"))
    first = min(t["date"] for t in trades)
    prices, names, missing = load_prices(trades, first)

    market_fx = None
    if any(t["ccy"] != "AUD" for t in trades):
        got = _download(["AUDUSD=X"], first)
        market_fx = got.get("AUDUSD=X")
        if market_fx is None:
            print("  AUD/USD market rate unavailable: using your latest conversion rate")
    fx = FX(cash, market_fx)

    out = build(trades, prices, names, missing, start, fx, cash)
    (DATA_DIR / "portfolio.json").write_text(json.dumps(out, separators=(",", ":")))

    s = out["summary"]
    print(f"Portfolio as of {out['priceDate']} (AUD): {s['openPositions']} open, "
          f"value ${s['marketValue']:,.2f} vs cost ${s['openCost']:,.2f} "
          f"(unrealised {s['unrealised']:+,.2f}); realised {s['realised']:+,.2f} "
          f"over {s['closedTrades']} sales; {len(out['weekly'])} weekly rows")
    if out["fx"]["audUsd"]:
        print(f"  AUD/USD {out['fx']['audUsd']} ({out['fx']['source']}); "
              f"USD cash US${out['cash']['usdCash']:,.2f}")
    if missing:
        print(f"  no price data for {', '.join(sorted(missing))}: valued at trade prices")
    print("Saved data/portfolio.json")


if __name__ == "__main__":
    sys.exit(main())
