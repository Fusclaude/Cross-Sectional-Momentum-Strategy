# Integrating into Cross-Sectional-Momentum-Strategy

Ordered so that nothing is committed until you've seen it work locally.

## 1. Branch

```bash
git checkout -b quant-research-layer
```

## 2. Copy files in

```
config.yaml                          → repo root          (new)
requirements.txt                     → repo root          (REPLACES existing)
METHODOLOGY.md                       → repo root          (new)
.gitignore                           → repo root          (new — merge if you have one)
scripts/metrics.py                   → scripts/           (new)
scripts/panel.py                     → scripts/           (new)
scripts/evaluation.py                → scripts/           (new)
scripts/portfolio.py                 → scripts/           (new)
scripts/build_analytics.py           → scripts/           (new)
scripts/run_signals.py               → scripts/           (REPLACES existing)
scripts/fetch_prices.py              → scripts/           (REPLACES existing)
scripts/picks_history_writer.py      → scripts/           (unchanged, already yours)
tests/test_engine.py                 → tests/             (new)
tearsheet.html                       → docs/              (new — note: docs/, not root)
.github/workflows/refresh.yml        → .github/workflows/ (REPLACES existing)
```

`fetch_universe.py` and `docs/index.html` are untouched.

`tearsheet.html` goes in `docs/` because that's what Pages serves, and it
fetches `data/analytics.json` relative to itself — which resolves to
`docs/data/analytics.json`, exactly where the workflow copies it.

## 3. Clean the duplicate history rows

Your `picks_history.jsonl` has 14 rows across 6 unique dates. The new writer is
idempotent so it won't happen again, but the existing duplicates need one pass:

```bash
python scripts/picks_history_writer.py merge \
  data/picks_history.jsonl data/picks_history.jsonl
wc -l data/picks_history.jsonl     # expect 6
```

## 4. Run it locally first

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # expect 21 passed
python scripts/fetch_universe.py
python scripts/fetch_prices.py      # ~5-10 min: 10y history for 800 tickers
python scripts/run_signals.py
python scripts/build_analytics.py
```

Then look at it:

```bash
mkdir -p docs/data && cp data/*.json data/*.jsonl docs/data/
cd docs && python -m http.server 8000
# open http://localhost:8000/tearsheet.html
# and    http://localhost:8000/index.html   (confirm the old dashboard still works)
```

`file://` will not work — the browser blocks the fetch.

## 5. Repo settings

- **Settings → Pages → Source: GitHub Actions.** If it currently says "Deploy
  from a branch", switch it, otherwise `deploy-pages` fails.
- **Settings → Actions → General → Workflow permissions: Read and write.**
  The workflow commits data back to the repo.

No secrets or API keys needed — yfinance and Wikipedia are both unauthenticated.

## 6. Merge and trigger

```bash
git add -A && git commit -m "Add research analytics layer"
git push -u origin quant-research-layer
```

Open a PR, merge, then **Actions → Refresh Momentum Signals → Run workflow** to
trigger the first run manually rather than waiting for Saturday.

## 7. Link the tearsheet from the dashboard

Add to the bottom nav in `docs/index.html` (inside `<nav class="bottom-nav">`):

```html
<a class="nav-item" href="tearsheet.html" style="text-decoration:none">
  <svg viewBox="0 0 24 24"><path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zM9 17H7v-5h2v5zm4 0h-2V7h2v10zm4 0h-2v-7h2v7z"/></svg>
  <span>Research</span>
</a>
```

---

## What changes on the first run

**`lookback_days` is 3650, not 760.** The first fetch pulls ~10 years and takes
noticeably longer. This is the single most important change in the whole set —
every statistic on the tearsheet is currently computed on 53 usable
cross-sections, and 10 years takes that to ~450.

**Expect the numbers to move, possibly a lot.** More history means the S&P IC of
0.042 and the ASX IC of −0.0007 both get re-estimated against a much larger
sample that includes at least one real drawdown. Neither current figure should
be treated as a prior.

**Price panels are now gitignored.** They're ~7 MB per run once 10y history and
volume are enabled; committing them weekly would add ~350 MB of git history per
year for files nothing reads between runs. The pipeline fetches them in the same
job that consumes them. `sp500_base.json` and `asx300_base.json` are still
committed — those are inputs, and the S&P one is what rescues the run when
Wikipedia blocks the runner.

**The dashboard keeps working untouched.** All 8 legacy factors reconcile
bit-exactly with what `index.html` reads today; everything new is additive
(`stocks[].risk`, `markets[].book`, `manifest`).

## If the first run fails

| Failure | Meaning |
|---|---|
| `pytest` step fails | A lookahead or arithmetic regression. Don't skip it — that gate is the reason it runs before the fetch. |
| `DATA QUALITY GATE FAILED` | Stale feed, suspected unadjusted split, or a latest bar >10 days old. The message names the market and the specific check. |
| `SANITY CHECK FAILED` | Outputs built but look wrong (too few names, empty book, config mismatch between signals and analytics). |
| `deploy-pages` fails | Pages source is still set to "Deploy from a branch". |
| S&P fetch falls back | Wikipedia blocked the runner. Usually transient; the cached list is used and the dashboard flags it. |

All four gates fail the job *before* anything is committed, which is the intent:
bad data producing a plausible-looking top 5 is worse than a crash, because
nobody investigates a number that looks fine.
