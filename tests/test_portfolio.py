"""
test_portfolio.py
─────────────────
The Portfolio tab reports real money, so the matching and valuation rules are
pinned down on small trade lists where the answer can be worked by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_portfolio as bp  # noqa: E402


def tr(date, ticker, side, shares, amount, market="ASX"):
    code = ticker.split(".")[0]
    return {"date": pd.Timestamp(date), "code": code, "market": market,
            "ticker": bp.key(code, market), "side": side, "shares": shares,
            "amount": amount, "ccy": "AUD" if market == "ASX" else "USD"}


def fx_cash(*conversions):
    """FX events: (date, aud, usd)."""
    return [{"date": pd.Timestamp(d), "type": "FX", "aud": a, "usd": u} for d, a, u in conversions]


def test_fifo_sells_oldest_lots_first():
    trades = [tr("2026-01-02", "AAA", "BUY", 10, 100.0),    # $10.00
              tr("2026-02-02", "AAA", "BUY", 10, 200.0),    # $20.00
              tr("2026-03-02", "AAA", "SELL", 15, 270.0)]
    closed, lots, _ = bp.match_fifo(trades)
    assert closed[0]["cost"] == pytest.approx(100.0 + 5 * 20.0)
    assert closed[0]["pnl"] == pytest.approx(70.0)
    assert closed[0]["firstBuy"] == pd.Timestamp("2026-01-02")
    assert [lot[1] for lot in lots["AAA"]] == [5]


def test_overselling_is_an_error():
    with pytest.raises(ValueError):
        bp.match_fifo([tr("2026-01-02", "AAA", "BUY", 5, 50.0),
                       tr("2026-01-09", "AAA", "SELL", 6, 60.0)])


def test_load_trades_skips_comments_and_sorts(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("# note\ndate,ticker,side,shares,amount\n"
                 "2026-02-01,bbb,sell,1,5\n2026-01-01,BBB,Buy,1,4\n")
    t = bp.load_trades(f)
    assert [x["side"] for x in t] == ["BUY", "SELL"]
    assert t[0]["ticker"] == "BBB" and t[0]["ccy"] == "AUD"


def test_load_trades_us_fractional_and_dividend(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("date,ticker,market,side,shares,amount,currency\n"
                 "2026-01-01,iren,US,BUY,0.5,10,USD\n2026-01-02,IREN,US,DIVIDEND,,0.03,USD\n")
    t = bp.load_trades(f)
    assert t[0]["ticker"] == "IREN.US" and t[0]["shares"] == 0.5
    assert t[1]["side"] == "DIVIDEND" and t[1]["shares"] == 0.0


def test_us_cost_uses_your_conversion_rate_and_sale_uses_market():
    fx = bp.FX(fx_cash(("2026-01-01", 1000.0, 700.0),      # 0.70 USD per AUD
                       ("2026-03-01", 1000.0, 600.0)),     # 0.60, after the buy
               market=pd.Series({pd.Timestamp("2026-03-27"): 0.80}))
    trades = [tr("2026-01-05", "XYZ", "BUY", 0.25, 70.0, "US"),
              tr("2026-03-27", "XYZ", "SELL", 0.25, 80.0, "US")]
    closed, lots, _ = bp.match_fifo(trades, fx)
    c = closed[0]
    assert c["pnl"] == pytest.approx(10.0)                   # in USD
    assert c["costAud"] == pytest.approx(70.0 / 0.70)        # rate paid on 1 Jan
    assert c["proceedsAud"] == pytest.approx(80.0 / 0.80)    # market rate at sale
    assert c["pnlAud"] == pytest.approx(0.0)                 # the AUD rally ate the gain
    assert not lots["XYZ.US"]


def test_grant_has_zero_cost_and_dividends_count_as_realised():
    idx = pd.date_range("2026-01-02", periods=2, freq="W-FRI")
    prices = pd.DataFrame({"NV.US": [100.0, 120.0]}, index=idx)
    fx = bp.FX(fx_cash(("2026-01-01", 100.0, 50.0)))         # 0.50
    trades = [tr("2026-01-01", "NV", "GRANT", 0.1, 0.0, "US"),
              tr("2026-01-06", "NV", "DIVIDEND", 0.0, 1.0, "US")]
    out = bp.build(trades, prices, {}, set(), idx[0], fx, [])
    s = out["summary"]
    assert out["open"][0]["costAud"] == 0.0
    assert s["marketValue"] == pytest.approx(0.1 * 120.0 / 0.5)
    assert s["dividends"] == pytest.approx(2.0) and s["realised"] == pytest.approx(2.0)
    assert s["closedTrades"] == 0


def test_cash_summary_tracks_uninvested_usd():
    cash = fx_cash(("2026-01-01", 1000.0, 700.0)) + [
        {"date": pd.Timestamp("2026-01-01"), "type": "DEPOSIT", "aud": 1200.0, "usd": 0.0}]
    trades = [tr("2026-01-02", "A", "BUY", 1.0, 500.0, "US"),
              tr("2026-01-09", "A", "SELL", 1.0, 450.0, "US"),
              tr("2026-01-10", "B", "BUY", 0.3, 600.0, "US")]
    cs = bp.cash_summary(cash, trades)
    assert cs["depositedAud"] == 1200.0
    assert cs["usdCash"] == pytest.approx(700.0 - 500.0 + 450.0 - 600.0)


def test_weekly_history_values_and_realised():
    idx = pd.date_range("2026-01-02", periods=4, freq="W-FRI")
    prices = pd.DataFrame({"AAA": [10.0, 12.0, None, 15.0]}, index=idx)
    trades = [tr("2026-01-01", "AAA", "BUY", 10, 100.0),
              tr("2026-01-13", "AAA", "SELL", 4, 50.0),       # cost 40, +10
              tr("2026-01-13", "ZZZ", "BUY", 2, 30.0)]        # no prices at all
    rows = bp.weekly_history(trades, prices, idx[0])
    w0, w2, w3 = rows[0], rows[2], rows[3]
    assert w0["marketValue"] == 100.0 and w0["unrealised"] == 0.0
    assert w0["weekChange"] is None
    # week 3: 6 AAA at a carried-forward $12, 2 ZZZ at the $15 trade price
    assert w2["marketValue"] == pytest.approx(6 * 12.0 + 2 * 15.0)
    assert w2["realisedToDate"] == pytest.approx(10.0)
    assert w2["estimated"] == ["ZZZ"]
    assert w3["openCost"] == pytest.approx(60.0 + 30.0)
    assert w3["netInvested"] == pytest.approx(100.0 - 50.0 + 30.0)
    assert w3["totalPnl"] == pytest.approx(w3["unrealised"] + 10.0)


def test_build_summary_ties_out():
    idx = pd.date_range("2026-01-02", periods=2, freq="W-FRI")
    prices = pd.DataFrame({"AAA": [10.0, 11.0], "BBB": [5.0, 4.0]}, index=idx)
    trades = [tr("2026-01-01", "AAA", "BUY", 10, 100.0),
              tr("2026-01-01", "BBB", "BUY", 10, 50.0),
              tr("2026-01-08", "BBB", "SELL", 10, 40.0)]
    out = bp.build(trades, prices, {}, set(), idx[0])
    s = out["summary"]
    assert s["marketValue"] == pytest.approx(110.0)
    assert s["unrealised"] == pytest.approx(10.0)
    assert s["realised"] == pytest.approx(-10.0)
    assert s["totalPnl"] == pytest.approx(0.0)
    assert out["weekly"][-1]["totalPnl"] == pytest.approx(s["totalPnl"])


def test_manual_prices_fill_gaps_but_never_override_market():
    idx = pd.date_range("2026-01-02", periods=4, freq="W-FRI")
    prices = pd.DataFrame({"AAA": [None, 10.0, 11.0, 12.0]}, index=idx)
    manual = [{"date": pd.Timestamp("2026-01-01"), "ticker": "AAA", "price": 9.0},
              {"date": pd.Timestamp("2026-01-20"), "ticker": "AAA", "price": 99.0},   # market exists
              {"date": pd.Timestamp("2026-01-09"), "ticker": "NEW.US", "price": 5.0},
              {"date": pd.Timestamp("2026-01-30"), "ticker": "NEW.US", "price": 6.0}]  # after last bar
    filled, mask = bp.apply_manual(prices, manual)
    assert list(filled["AAA"]) == [9.0, 10.0, 11.0, 12.0]
    assert list(mask["AAA"]) == [True, False, False, False]
    assert pd.isna(filled["NEW.US"].iloc[0])                 # not back-dated
    assert list(filled["NEW.US"].iloc[1:]) == [5.0, 5.0, 6.0]  # latest week takes the newest entry
    assert mask["NEW.US"].iloc[1:].all()


def test_manual_fx_applies_from_its_date_only():
    fx = bp.FX(fx_cash(("2026-01-01", 100.0, 70.0)),
               manual=pd.Series({pd.Timestamp("2026-02-01"): 0.60}))
    assert fx.market(pd.Timestamp("2026-01-15")) == pytest.approx(0.70)
    assert fx.market(pd.Timestamp("2026-02-06")) == pytest.approx(0.60)
    assert fx.source == "manual"


def test_value_reports_price_source():
    idx = pd.date_range("2026-01-02", periods=1, freq="W-FRI")
    prices = pd.DataFrame({"AAA": [10.0]}, index=idx)
    trades = [tr("2026-01-01", "AAA", "BUY", 1, 8.0), tr("2026-01-01", "BBB", "BUY", 1, 4.0),
              tr("2026-01-01", "CCC", "BUY", 1, 2.0)]
    filled, mask = bp.apply_manual(prices, [{"date": idx[0], "ticker": "BBB", "price": 5.0}])
    out = bp.build(trades, filled, {}, {"CCC"}, idx[0], None, [], mask)
    src = {p["ticker"]: p["priceSource"] for p in out["open"]}
    assert src == {"AAA": "market", "BBB": "manual", "CCC": "trade"}
    assert out["weekly"][0]["manual"] == ["BBB"] and out["weekly"][0]["estimated"] == ["CCC"]


def test_load_manual(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("# x\ndate,ticker,market,price\n2026-01-02,iren,US,50\n2026-01-02,AUDUSD,,0.66\n"
                 "2026-01-02,CDA,,60\n")
    m = bp.load_manual(f)
    assert [x["ticker"] for x in m] == ["IREN.US", "AUDUSD", "CDA"]


def test_weekly_flows_and_benchmarks():
    idx = pd.date_range("2026-01-02", periods=3, freq="W-FRI")
    prices = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=idx)
    trades = [tr("2026-01-01", "AAA", "BUY", 10, 100.0),
              tr("2026-01-07", "AAA", "BUY", 5, 55.0),
              tr("2026-01-14", "AAA", "SELL", 3, 36.0),
              tr("2026-01-14", "AAA", "DIVIDEND", 0, 2.0)]
    bench = {"IDX": pd.Series([1000.0, 1010.0, 990.0], index=idx)}
    rows = bp.weekly_history(trades, prices, idx[0], None, None, bench)
    assert rows[0]["flowAud"] is None
    assert rows[1]["flowAud"] == pytest.approx(55.0)
    assert rows[2]["flowAud"] == pytest.approx(-36.0 - 2.0)   # sale and dividend both leave
    assert [r["benchmarks"]["IDX"] for r in rows] == [1000.0, 1010.0, 990.0]
