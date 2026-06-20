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
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def sp500_tickers() -> list[str]:
    """Current S&P 500 constituents from Wikipedia's maintained table."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    names = dict(zip(tickers, df["Security"].astype(str)))
    return tickers, names


def asx300_tickers() -> tuple[list[str], dict]:
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
    return tickers, names


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Fetching S&P 500 constituents...")
    sp_tickers, sp_names = sp500_tickers()
    print(f"  {len(sp_tickers)} tickers")

    print("Loading ASX 300 base list...")
    asx_tickers, asx_names = asx300_tickers()
    print(f"  {len(asx_tickers)} tickers")

    universe = {
        "sp500": {"tickers": sp_tickers, "names": sp_names},
        "asx300": {"tickers": asx_tickers, "names": asx_names},
    }
    out_path = DATA_DIR / "universe.json"
    with open(out_path, "w") as f:
        json.dump(universe, f)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
