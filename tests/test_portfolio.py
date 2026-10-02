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


def tr(date, ticker, side, shares, amount):
    return {"date": pd.Timestamp(date), "ticker": ticker, "side": side,
            "shares": shares, "amount": amount}


def test_fifo_sells_oldest_lots_first():
    trades = [tr("2026-01-02", "AAA", "BUY", 10, 100.0),    # $10.00
              tr("2026-02-02", "AAA", "BUY", 10, 200.0),    # $20.00
              tr("2026-03-02", "AAA", "SELL", 15, 270.0)]
    closed, lots = bp.match_fifo(trades)
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
    assert t[0]["ticker"] == "BBB"


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
