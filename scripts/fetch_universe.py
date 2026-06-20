"""
fetch_universe.py
─────────────────
Pulls the CURRENT constituent list for the S&P 500 and ASX 300.
This solves the "index membership drifts every quarter" problem —
instead of using a list you typed once, this re-derives it every run.

S&P 500: scraped from Wikipedia's maintained constituent table (the standard
         free source; updated whenever S&P actually changes the index).
ASX 300: ASX doesn't publish a free machine-readable list, so this uses
         your last-known ASX 300 ticker file as a base and just refreshes
         prices for those. If you have a paid data source for ASX
         membership later, swap out asx_tickers() only.
"""
import requests
import pandas as pd
import json
import io
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def sp500_tickers() -> tuple[list[str], dict, dict, bool]:
    """
    Current S&P 500 constituents from Wikipedia's maintained table.

    Wikipedia (or a CDN in front of it) occasionally serves a block/CAPTCHA
    page instead of the article to traffic from shared CI IP ranges like
    GitHub Actions runners. We detect that case explicitly and fall back
    to the last successfully-fetched list rather than crashing the whole
    pipeline over a transient block.

    Returns (tickers, names, sectors, used_fallback) — used_fallback is
    surfaced all the way through to the dashboard so a silent stale-data
    situation is never actually silent to you. Sectors come from
    Wikipedia's "GICS Sector" column.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    fallback_path = DATA_DIR / "sp500_base.json"

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        if "<table" not in resp.text.lower():
            raise ValueError(
                "Response did not contain any HTML tables — likely a "
                "block/CAPTCHA page rather than the real article. "
                f"First 200 chars: {resp.text[:200]!r}"
            )

        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        names = dict(zip(tickers, df["Security"].astype(str)))
        sector_col = "GICS Sector" if "GICS Sector" in df.columns else None
        sectors = dict(zip(tickers, df[sector_col].astype(str))) if sector_col else {}

        # Cache this successful pull so a future block has something to fall
        # back to instead of failing the whole run.
        with open(fallback_path, "w") as f:
            json.dump({"tickers": tickers, "names": names, "sectors": sectors}, f)

        return tickers, names, sectors, False

    except Exception as e:
        print(f"  WARNING: live S&P 500 fetch failed ({e})")
        if fallback_path.exists():
            print(f"  Falling back to last cached list: {fallback_path}")
            with open(fallback_path) as f:
                cached = json.load(f)
            return cached["tickers"], cached["names"], cached.get("sectors", {}), True
        raise RuntimeError(
            "S&P 500 fetch failed and no cached fallback exists yet "
            "(this only happens on a clean repo's very first run). "
            "Re-run the workflow — Wikipedia blocks are usually transient. "
            "If it keeps failing, see the README's 'S&P 500 fetch troubleshooting' "
            "section for an alternative data source."
        ) from e


def asx300_tickers() -> tuple[list[str], dict, dict]:
    """
    ASX 300 constituents. No free, reliably-updated machine-readable source
    exists, so we maintain a base list (data/asx300_base.json) which you
    refresh manually every quarter when ASX rebalances (4x/year — much less
    frequent than this pipeline runs). This is the ONE part of the system
    that still needs occasional manual input, and it's a 2-minute job:
    copy the new list from the ASX/S&P website into asx300_base.json.
    """
    base_file = DATA_DIR / "asx300_base.json"
    if not base_file.exists():
        raise FileNotFoundError(
            f"{base_file} not found. Create it with your current ASX 300 "
            f"ticker + name list (see asx300_base.example.json for the format)."
        )
    with open(base_file) as f:
        base = json.load(f)
    tickers = [t + ".AX" for t in base["tickers"]]
    names = {t + ".AX": n for t, n in base["names"].items()}
    sectors = {t + ".AX": s for t, s in base.get("sectors", {}).items()}
    return tickers, names, sectors


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Fetching S&P 500 constituents...")
    sp_tickers, sp_names, sp_sectors, sp_used_fallback = sp500_tickers()
    print(f"  {len(sp_tickers)} tickers" + ("  [USED CACHED FALLBACK]" if sp_used_fallback else ""))

    print("Loading ASX 300 base list...")
    asx_tickers, asx_names, asx_sectors = asx300_tickers()
    print(f"  {len(asx_tickers)} tickers")

    universe = {
        "sp500": {
            "tickers": sp_tickers,
            "names": sp_names,
            "sectors": sp_sectors,
            "usedFallback": sp_used_fallback,
        },
        "asx300": {
            "tickers": asx_tickers,
            "names": asx_names,
            "sectors": asx_sectors,
            "usedFallback": False,
        },
    }
    out_path = DATA_DIR / "universe.json"
    with open(out_path, "w") as f:
        json.dump(universe, f)
    print(f"Saved {out_path}")
    if sp_used_fallback:
        print(
            "\n⚠ NOTE: S&P 500 list came from the cached fallback this run, "
            "not a fresh Wikipedia fetch. This will be flagged on the dashboard."
        )


if __name__ == "__main__":
    main()
