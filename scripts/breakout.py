"""
breakout.py
───────────
All-time-high (ATH) breakout detection. Pure functions, no I/O.

WHAT THIS IS, AND HOW IT DIFFERS FROM THE REST OF THE REPO
The momentum engine is CROSS-SECTIONAL: it ranks every stock against every
other stock and buys the relative winners, whatever the absolute level. This
module is EVENT-DRIVEN and absolute: a stock qualifies when its weekly close
clears the highest close it has ever printed. Practitioners know it as the
Darvas box / CAN SLIM "N" (new highs out of a base) / Donchian breakout with an
unbounded lookback; the closest academic evidence is George & Hwang (2004) on
52-week highs and Li & Yu (2012) on nearness to the historical high.

DEFINITIONS (all on weekly closes, all point-in-time)
  ath            highest close at or before week t
  breakout       close at t strictly above the highest close before t, after a
                 base: the previous high was set >= min_base_weeks earlier.
                 A stock grinding to a new high every week is a continuation,
                 not a breakout, and does not reset the clock.
  breakout level the old ATH that was cleared — the pivot the trade is judged
                 against. Closing more than stop_pct below it is a failed
                 breakout, and the setup is dead until a new breakout forms.
  extension      how far above the pivot the price is now. Past max_extension
                 the move has already happened; buying there is chasing.

"All-time" means all-time within the price file. The pipeline fetches
data.lookback_days (~10y), so for a name whose true peak predates that window
this is a 10-year high, not an ATH. min_history_weeks guards the other end:
a stock listed 30 weeks ago makes an "ATH" most weeks and it means nothing.

LOOKAHEAD: every output at row t uses closes at or before t only. The test
suite rebuilds the state panel from truncated history and asserts no value
changes.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

PPY = 52

STATE_KEYS = ["ath", "pct_from_ath", "weeks_since_breakout", "breakout_level",
              "base_weeks", "breakout_volume_ratio", "failed", "candidate"]


def breakout_state(prices: pd.DataFrame, bcfg: dict,
                   dollar_volume: pd.DataFrame | None = None,
                   min_price: float = 0.0) -> dict[str, pd.DataFrame]:
    """
    Walk each ticker forward in time and record its breakout state at every
    week. Returns {key: (dates x tickers) DataFrame} for every key in
    STATE_KEYS. `candidate` is the buy-list mask: a qualifying breakout within
    the last recent_weeks that has neither failed nor run too far.

    A per-ticker loop rather than a vectorised expression, on purpose: the
    state (which pivot, has it failed since) is path-dependent, and a loop over
    ~500 x 520 cells runs in well under a second and reads like the rule.
    """
    min_hist = int(bcfg["min_history_weeks"])
    min_base = int(bcfg["min_base_weeks"])
    recent = int(bcfg["recent_weeks"])
    stop = float(bcfg["stop_pct"])
    max_ext = float(bcfg["max_extension"])
    vol_w = int(bcfg.get("volume_lookback_weeks", 50))

    px = prices.to_numpy(dtype=float)
    T, N = px.shape
    dv = None
    if dollar_volume is not None:
        dv = dollar_volume.reindex(index=prices.index, columns=prices.columns).to_numpy(dtype=float)

    out = {k: np.full((T, N), np.nan) for k in STATE_KEYS}

    for j in range(N):
        ath = -math.inf
        last_high = -1          # row index where the current ATH was set
        n_bars = 0              # valid closes seen BEFORE the current row
        bo_idx, bo_level, bo_base, bo_vr = -1, math.nan, math.nan, math.nan
        failed = False
        for i in range(T):
            p = px[i, j]
            if not np.isfinite(p) or p <= 0:
                continue
            if p > ath:
                if n_bars >= min_hist and np.isfinite(ath) and i - last_high >= min_base:
                    bo_idx, bo_level, bo_base, failed = i, ath, i - last_high, False
                    bo_vr = math.nan
                    if dv is not None and i >= 1:
                        hist = dv[max(0, i - vol_w):i, j]
                        hist = hist[np.isfinite(hist)]
                        if len(hist) >= 10 and np.isfinite(dv[i, j]) and np.median(hist) > 0:
                            bo_vr = float(dv[i, j] / np.median(hist))
                ath, last_high = p, i
            n_bars += 1

            if bo_idx >= 0 and p < bo_level * (1.0 - stop):
                failed = True

            out["ath"][i, j] = ath
            out["pct_from_ath"][i, j] = p / ath - 1.0
            if bo_idx >= 0:
                wsb = i - bo_idx
                out["weeks_since_breakout"][i, j] = wsb
                out["breakout_level"][i, j] = bo_level
                out["base_weeks"][i, j] = bo_base
                out["breakout_volume_ratio"][i, j] = bo_vr
                out["failed"][i, j] = float(failed)
                out["candidate"][i, j] = float(
                    wsb <= recent and not failed and p >= min_price
                    and p <= bo_level * (1.0 + max_ext))
            else:
                out["candidate"][i, j] = 0.0

    return {k: pd.DataFrame(v, index=prices.index, columns=prices.columns)
            for k, v in out.items()}


def rs_rating(prices: pd.DataFrame, weeks: int = 26) -> pd.DataFrame:
    """
    Relative-strength rating, 1-99: percentile of the trailing `weeks`-week
    return within that week's cross-section. Used only to ORDER candidates —
    when forty names break out in the same week you still have to pick.
    """
    ret = prices / prices.shift(weeks) - 1.0
    return (ret.rank(axis=1, pct=True) * 98 + 1).round()


def event_study(state: dict[str, pd.DataFrame], prices: pd.DataFrame,
                horizons: list[int]) -> dict:
    """
    Forward returns after every historical breakout, measured from the
    breakout week's close, against the equal-weight universe over the same
    window. Excess is per-event, so a bull market lifting everything does not
    count as edge.

    Overlapping events (same stock, nearby breakouts; many stocks breaking
    out in the same rally) are not independent, so the n here overstates the
    effective sample size. Read the hit rate and median, not a t-stat.
    """
    is_bo = state["weeks_since_breakout"] == 0
    ew = prices.pct_change().mean(axis=1)
    ew_level = (1.0 + ew.fillna(0.0)).cumprod()
    out = {"n_events": int(is_bo.sum().sum()), "horizons": {}}
    for h in horizons:
        fwd = prices.shift(-h) / prices - 1.0
        bench = ew_level.shift(-h) / ew_level - 1.0
        r = fwd[is_bo].stack()
        if r.empty:
            continue
        b = bench.reindex(r.index.get_level_values(0)).to_numpy()
        x = r.to_numpy() - b
        ok = np.isfinite(x)
        r, x = r.to_numpy()[ok], x[ok]
        if len(r) == 0:
            continue
        out["horizons"][str(h)] = {
            "n": int(len(r)),
            "mean": float(np.mean(r)),
            "median": float(np.median(r)),
            "hit_rate": float(np.mean(r > 0)),
            "mean_excess": float(np.mean(x)),
            "median_excess": float(np.median(x)),
            "excess_hit_rate": float(np.mean(x > 0)),
        }
    return out


def backtest(state: dict[str, pd.DataFrame], prices: pd.DataFrame, bcfg: dict,
             rs: pd.DataFrame, cost_bps: float = 0.0) -> dict:
    """
    Trade the rule as written. At each weekly close:
      exit  any holding whose close is below its entry pivot * (1 - stop_pct),
            or more than trailing_stop below its highest close since entry;
      enter candidates (highest RS first) until max_positions are held.
    Holdings are equal-weighted and earn the NEXT week's return, so there is
    no same-bar lookahead. Costs are charged on entries and exits.

    Every round trip is recorded in `tradeLog` (fills at the weekly close,
    before costs); positions still held at the last bar have exit None.
    """
    rets = prices.pct_change()
    cand = state["candidate"].fillna(0.0).astype(bool)
    level = state["breakout_level"]
    stop, trail = float(bcfg["stop_pct"]), float(bcfg["trailing_stop"])
    max_pos = int(bcfg["max_positions"])
    idx = prices.index

    held: dict[str, dict] = {}   # ticker -> {"pivot", "peak", "entry_date", "entry_px"}
    port, bench, n_held, trades, log, dates = [], [], [], 0, [], []
    started = False
    # The last bar is traded too (so this week's buys and sells appear in the
    # log) but earns no return: there is no next week yet.
    for i in range(len(idx)):
        dt = idx[i]
        row = prices.iloc[i]
        before = set(held)
        for t in list(held):
            p = row[t]
            if not np.isfinite(p):
                continue
            h = held[t]
            h["peak"] = max(h["peak"], p)
            reason = ("stop" if p < h["pivot"] * (1.0 - stop) else
                      "trailing_stop" if p < h["peak"] * (1.0 - trail) else None)
            if reason:
                log.append(_trade(t, held.pop(t), dt, float(p), reason))
        if len(held) < max_pos:
            new = [t for t in cand.columns[cand.iloc[i].to_numpy()] if t not in held]
            new.sort(key=lambda t: -(rs.at[dt, t] if np.isfinite(rs.at[dt, t]) else -1))
            for t in new[:max_pos - len(held)]:
                held[t] = {"pivot": float(level.at[dt, t]), "peak": float(row[t]),
                           "entry_date": dt, "entry_px": float(row[t])}
        turnover = len(before ^ set(held))
        trades += turnover
        if i == len(idx) - 1 or (not held and not started):
            continue
        started = True
        nxt = idx[i + 1]
        w = 1.0 / len(held) if held else 0.0
        r = float(rets.loc[nxt].reindex(list(held)).fillna(0.0).sum() * w)
        # Turnover in weight terms: each name entered or exited is ~1/n of book.
        r -= turnover * w * cost_bps / 10000.0 if held else 0.0
        port.append(r)
        bench.append(float(rets.loc[nxt].mean()))
        n_held.append(len(held))
        dates.append(nxt)

    for t, h in held.items():
        log.append(_trade(t, h, None, float(prices[t].dropna().iloc[-1]), "open"))

    if len(port) < 26:
        return {"error": "insufficient history for backtest", "n_periods": len(port)}
    return {
        "strategy": perf(np.array(port)),
        "equal_weight_universe": perf(np.array(bench)),
        "avg_positions": float(np.mean(n_held)),
        "pct_weeks_in_cash": float(np.mean(np.array(n_held) == 0)),
        "trades": int(trades),
        "tradeLog": log,
        # weekly returns, dated by the week they were earned
        "series": pd.DataFrame({"strategy": port, "universe": bench,
                                "positions": n_held}, index=pd.DatetimeIndex(dates)),
    }


def _trade(t: str, h: dict, exit_date, exit_px: float, reason: str) -> dict:
    return {"ticker": t, "entry_date": h["entry_date"], "entry_px": h["entry_px"],
            "pivot": h["pivot"], "exit_date": exit_date, "exit_px": exit_px,
            "return": exit_px / h["entry_px"] - 1.0, "reason": reason}


def perf(pr: np.ndarray) -> dict:
    curve = np.cumprod(1.0 + pr)
    years = len(pr) / PPY
    cagr = float(curve[-1] ** (1.0 / years) - 1.0) if curve[-1] > 0 else -1.0
    sd = float(np.std(pr, ddof=1))
    return {
        "years": round(years, 2),
        "cagr": cagr,
        "vol_ann": sd * math.sqrt(PPY),
        "sharpe": float(np.mean(pr) / sd * math.sqrt(PPY)) if sd > 1e-12 else None,
        "max_drawdown": float(np.min(curve / np.maximum.accumulate(curve) - 1.0)),
    }
