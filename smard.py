"""
smard.py - download hourly series from SMARD.de, the electricity-market data platform of
the German Federal Network Agency (Bundesnetzagentur). Free, no API key.

SMARD stores each series in weekly JSON files. This module downloads them once into
data/cache/ and builds one tidy hourly table, data/smard_hourly.csv.gz.

Data licence: CC BY 4.0, "Bundesnetzagentur | SMARD.de".

Usage:
    python smard.py            # download / update, then write data/smard_hourly.csv.gz
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.smard.de/app/chart_data"
HEADERS = {"User-Agent": "german-electricity-price-forecasting (portfolio project)"}
DATA = Path(__file__).resolve().parent / "data"
CACHE = DATA / "cache"
START = pd.Timestamp("2018-10-01", tz="UTC")   # the DE-LU price zone starts here

# column name -> (SMARD filter id, region)
SERIES = {
    "price": (4169, "DE-LU"),            # day-ahead price, EUR/MWh
    "load": (410, "DE"),                 # actual grid load, MWh
    "fc_wind_onshore": (123, "DE"),      # day-ahead forecasts of generation, MWh
    "fc_wind_offshore": (3791, "DE"),
    "fc_solar": (125, "DE"),
    "fc_total_generation": (122, "DE"),
}


def _get(url):
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def week_starts(filter_id, region):
    return _get(f"{BASE}/{filter_id}/{region}/index_hour.json")["timestamps"]


def fetch_week(filter_id, region, ts, refresh=False):
    """One weekly file, cached on disk. Weeks that may still change (the latest two)
    are re-downloaded when refresh is True."""
    path = CACHE / f"{filter_id}_{region}_{ts}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    series = _get(f"{BASE}/{filter_id}/{region}/{filter_id}_{region}_hour_{ts}.json")["series"]
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(series))
    time.sleep(0.05)   # be polite to the public API
    return series


def fetch_series(name):
    filter_id, region = SERIES[name]
    stamps = [t for t in week_starts(filter_id, region) if pd.Timestamp(t, unit="ms", tz="UTC") >= START - pd.Timedelta(days=7)]
    with ThreadPoolExecutor(max_workers=6) as pool:   # a few requests at a time, not a flood
        weeks = pool.map(lambda it: fetch_week(filter_id, region, it[1], refresh=it[0] >= len(stamps) - 2),
                         enumerate(stamps))
        rows = [point for week in weeks for point in week]
    s = pd.Series({pd.Timestamp(t, unit="ms", tz="UTC"): v for t, v in rows if v is not None}, name=name, dtype=float)
    return s[~s.index.duplicated()].sort_index()


def build_table():
    frames = []
    for name in SERIES:
        print(f"  {name} ...", flush=True)
        frames.append(fetch_series(name))
    df = pd.concat(frames, axis=1, sort=True)
    df = df[df.index >= START]
    df.index.name = "time_utc"
    return df


if __name__ == "__main__":
    table = build_table()
    out = DATA / "smard_hourly.csv.gz"
    table.to_csv(out, float_format="%.2f")
    print(f"Wrote {out}: {len(table):,} hours, {table.index.min()} to {table.index.max()}")
    print(table.notna().mean().round(3).to_string())
