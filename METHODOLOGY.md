# Methodology

How the signal is built, how it is evaluated, and — the part that matters most —
what the numbers are not allowed to be used for.

---

## 1. Pipeline

```
fetch_universe.py  →  fetch_prices.py  →  run_signals.py  →  build_analytics.py
   constituents        prices+volume       live cross-section    research tearsheet
                       + QA gate           + portfolio book      + IC / DSR / PBO
```

Every stage reads `config.yaml`. Nothing that affects a result is hard-coded in
a script. Each run writes a manifest containing the SHA-256 of the config, the
git revision, and digests of the input files, so any published pick can be
traced to the exact parameters and data that produced it.

| File | Role |
|---|---|
| `config.yaml` | Every tunable parameter. Single source of truth. |
| `scripts/metrics.py` | Pure metric functions. No I/O, fully unit-tested. |
| `scripts/panel.py` | Vectorised point-in-time factor panels. |
| `scripts/evaluation.py` | IC, decile spreads, turnover, DSR, PBO, backtest. |
| `scripts/portfolio.py` | Constrained construction and risk reporting. |
| `scripts/build_analytics.py` | Writes `data/analytics.json`. |
| `tearsheet.html` | Renders the research view. |
| `tests/test_engine.py` | 21 tests, including the lookahead detector. |

---

## 2. Factors

Sign convention: **higher is always better.** Factors whose natural direction is
inverted (volatility, drawdown, information discreteness) are negated at
construction. A sign-flipped factor still produces a plausible-looking ranking,
which is why this is enforced in one place and tested rather than tracked by
convention.

### Legacy set (unchanged, reconciles exactly against v1)

| Factor | Definition |
|---|---|
| `r121` `r91` `r61` `r31` | Skip-adjusted return: `P[t−1mo] / P[t−Nmo] − 1`. Skipping the last month removes short-term reversal contamination (Jegadeesh & Titman 1993). |
| `r2_12` `r2_9` `r2_6` `r2_3` | R² of a log-linear trend fit. Rewards stocks that trended smoothly rather than gapping. |

### Added

| Factor | Definition and rationale |
|---|---|
| `resid_mom` | Momentum on market-model residuals, standardised (Blitz, Huij & Martens 2011). Strips out the "winners were just high-beta to whatever ran" effect. Historically ~2× the risk-adjusted return of raw momentum with far smaller crashes. |
| `mom_vol_scaled` | 12-1 momentum ÷ realised vol (Barroso & Santa-Clara 2015). Momentum crashes cluster in high-vol states; scaling removes most of the negative skew. |
| `pct_52w_high` | Price ÷ 52-week high (George & Hwang 2004). More persistent than raw past return and less prone to buying parabolic moves. |
| `info_discreteness` | Frog-in-the-Pan (Da, Gurun & Warachka 2014): `sign(cum ret) × (%neg − %pos)`. Continuous information is underreacted to and persists; discrete information reverses. Negated. |
| `st_reversal` | 1-month reversal, kept as a separate sleeve rather than blended into momentum. |
| `low_vol` `low_beta` `idio_vol` | Rolling risk measures from a point-in-time market model. |
| `vol_regime` | 3m vol ÷ 12m vol. Expansion flags the state where momentum breaks. Negated. |
| `consistency` | Fraction of positive weeks over 12 months. |
| `sharpe_12m` `sortino_12m` `ulcer` `drawdown` | Trailing risk-adjusted performance and pain measures. |

### Composite

`winsorise (±3σ) → cross-sectional z-score → sector-neutralise → weighted sum → re-standardise`

Two deliberate departures from v1:

- **z-scores, not percentile ranks.** Ranks discard magnitude: a stock 4σ ahead
  of the field and one 0.1σ ahead both become "rank 1", so the composite cannot
  distinguish a strong month from a flat one.
- **sector neutralisation.** Without it the composite is largely a sector bet.
  On the current S&P cross-section the un-neutralised top 5 was five memory and
  storage stocks — one trade in five costumes.

---

## 3. Lookahead discipline

Every value at row *t* uses only prices at or before *t*. Forward returns are
produced in one place (`evaluation.forward_returns`) and are the only
forward-looking object in the system.

This is enforced mechanically, not by review. `test_point_in_time_truncation`
rebuilds the entire factor panel from truncated history at three cutoffs and
asserts that (a) no historical value changes, and (b) no value that was
computable with full history becomes NaN — because an extra NaN means the
factor needed data from the future.

The second check exists because the first version of this test **passed a
deliberately planted bug** (a centred rolling window): NaNs in the truncated
panel masked the mismatch. The detector has since been validated by planting
two separate lookahead bugs and confirming both are caught.

Rolling betas use a trailing window. Full-sample betas are a subtle and very
common source of lookahead in factor research.

---

## 4. Evaluation

| Statistic | What it answers | Bar used |
|---|---|---|
| **Rank IC** | Does the score correlate with *forward* return? | ≥ 0.03 |
| **Newey–West t-stat** | Is that IC distinguishable from zero, allowing for the autocorrelation that overlapping horizons create by construction? | ≥ 2.5 |
| **ICIR** | Is it *consistent*? Maps to strategy Sharpe via IR ≈ IC × √breadth. | ≥ 0.50 |
| **Decile monotonicity** | Does the effect hold across the whole cross-section, or only at the extreme? | ρ ≥ 0.70 |
| **Deflated Sharpe** | How much of the backtest Sharpe is explained by having tried many configurations? | ≥ 0.95 |
| **PBO (CSCV)** | Does in-sample rank predict out-of-sample rank, or is selection noise? | ≤ 0.30 |
| **Breakeven cost** | At what round-trip cost does the edge die? | ≥ 60bp |

Thresholds are written down in `tearsheet.html` **before** results are examined.
A threshold chosen after seeing the answer is not a threshold.

Two points that get skipped often enough to be worth stating:

- An OLS t-stat on overlapping forward returns is not slightly wrong; on a
  26-week horizon sampled weekly it can be off by roughly √h. Newey–West with a
  Bartlett kernel is used throughout.
- PBO above 0.5 means the selection procedure is *worse than random*. It is the
  single most useful number here, because it directly measures the thing a
  weight-slider UI encourages: searching until the backtest looks good.

---

## 5. Portfolio construction

Applied in order, each recorded per position as `binding_constraint` so every
weight is explainable:

1. **Liquidity** — position ≤ 5% of median daily dollar volume.
2. **Sector cap** — enforced at *selection*, by skipping names whose sector is
   full and filling further down the ranking.
3. **Position cap** — bounds single-name risk.
4. **Vol targeting** — gross exposure scaled to hit an ex-ante vol target using
   a Ledoit–Wolf shrunk covariance matrix.

Two bugs found and fixed while building this, both worth knowing about because
they are easy to reintroduce:

- Scaling the weights of an over-concentrated sector **does not** fix
  concentration. Five semiconductor stocks scaled down are still 100%
  semiconductors — just a smaller position. The cap only means something if it
  changes which names are held.
- Renormalising weights to sum to 1.0 after applying caps **silently reverses
  every cap**. If liquidity limits cut exposure to 30%, the other 70% is cash.

Cost model: `half-spread + commission + impact_coef × √participation`. Square-root
impact means cost scales with size, so the model knows a $50k and a $50m
position in the same microcap are not the same trade.

---

## 6. Known limitations

**Survivorship bias is present and not fixable from free data.** The universe is
today's index membership applied to all history. Delisted and demoted names are
absent. This inflates every backtest statistic — typically 1–4% p.a. for a
large-cap index, and materially more for a 300-name mid-cap index with real
turnover. Fixing it requires point-in-time constituent history.

**The ASX 300 list is manually maintained** and refreshed roughly quarterly, so
membership drifts between updates.

**Sample length.** With `lookback_days: 3650` there are ~450 usable weekly
cross-sections. At the previous 760 days there were ~53, and the standard error
on an IC estimate over 53 periods is roughly 0.14 — wider than any real equity
factor's IC. Results computed on the short sample are not conclusive, including
the ones that look good.

**Price-only.** No fundamentals, so no value, quality, profitability or accruals
sleeve. Everything here is a price-and-volume signal, which caps how much
diversification the composite can actually achieve.

**Weekly data.** Some effects (short-term reversal especially) are measured
poorly at weekly frequency.

---

## 7. Running it

```bash
pip install -r requirements.txt
python scripts/fetch_universe.py
python scripts/fetch_prices.py       # exits non-zero if the QA gate fails
python scripts/run_signals.py
python scripts/build_analytics.py
python -m pytest tests/ -q

python -m http.server 8000           # then open /tearsheet.html
```

The tearsheet fetches `data/analytics.json`, so it needs to be served over HTTP —
opening it via `file://` will be blocked by the browser.

---

## 8. What this is not

This is a measurement tool. It is designed to tell you when a signal is *not*
working, which is most of the time and is the useful half of the job.

It is not investment advice, not a recommendation to trade, and not a forecast.
Statistical relationships in markets routinely stop working, often precisely
when enough people find them. The correct response to a poor result here is not
to reweight until the backtest improves — that search process is exactly what
the deflated Sharpe and PBO figures exist to penalise.
