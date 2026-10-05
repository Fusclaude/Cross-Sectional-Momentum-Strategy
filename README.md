# Cross-Sectional-Momentum-Strategy
Cross Sectional Momentum Strategy for both the ASX300 and SNP500, run weekly. 

## Crypto sleeve

A separate cross-sectional momentum sleeve for crypto, with its own config,
workflow and dashboard (`docs/crypto/`, linked from the equity dashboard).

| Piece | File |
|---|---|
| Config (universe, factors, regime, costs) | `config_crypto.yaml` |
| Prices: Yahoo daily → closed Sunday-UTC weeks | `scripts/fetch_crypto.py` → `data/crypto_prices.json` |
| Factors, point-in-time universe, BTC regime | `scripts/crypto_engine.py` |
| Signals + research (IC, quintiles, backtest, DSR, PBO) | `scripts/run_crypto.py` → `data/crypto_signals.json` |
| Tests (lookahead truncation, stablecoin filter, JS/Python parity) | `tests/test_crypto.py` |
| Schedule | `.github/workflows/refresh_crypto.yml` — Mondays 00:45 UTC |

Key differences from the equity sleeve: the universe is rebuilt every week as
the top 50 coins by trailing dollar volume (no present-day market-cap list),
stablecoins are removed by name and by behaviour, factor windows are 4–26
weeks with no skip month, and a BTC-above-20-week-average regime filter is
evaluated alongside the unfiltered book.
