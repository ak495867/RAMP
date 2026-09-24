"""
RAMP Automated Real-World Data Ingestion Pipeline.
Fetches multi-asset market data from Yahoo Finance and macroeconomic series from St. Louis FRED.
Validates point-in-time correctness, aligns trading calendars, and stores in DuckDB & Parquet.
"""

import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import duckdb

from ramp.core.calendar import MarketCalendar
from ramp.data.collectors.yahoo import YahooDataCollector
from ramp.data.collectors.fred import FREDCollector
from ramp.data.validator import MarketDataValidator
from ramp.data.pit_store import PointInTimeStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Directories
PROJECT_ROOT = Path("d:/RAMP")
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_DIR = DATA_DIR / "parquet"
DUCKDB_PATH = DATA_DIR / "ramp_lakehouse.duckdb"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# 10-Asset Multi-Asset Liquid Universe + VIX
TARGET_SYMBOLS = [
    "SPY",      # S&P 500 (US Large Cap Equity)
    "QQQ",      # Nasdaq 100 (Tech / Growth Equity)
    "IWM",      # Russell 2000 (US Small Cap Equity)
    "EEM",      # MSCI Emerging Markets (Global Beta)
    "TLT",      # 20+ Year Treasury Bond (Long Duration / Deflation Hedge)
    "IEF",      # 7-10 Year Treasury Bond (Intermediate Duration)
    "GLD",      # SPDR Gold Trust (Inflation / Crisis Safe Haven)
    "DBC",      # Commodity Index Tracking Fund (Raw Materials / Cyclical Inflation)
    "UUP",      # US Dollar Index Bullish Fund (FX Reserve Beta)
    "BTC-USD",  # Bitcoin (Liquid Digital Asset / Asymmetric Liquidity Beta)
    "^VIX",     # CBOE Volatility Index (Implied Volatility Benchmark)
]

FRED_MACRO_SERIES = [
    "YIELD_CURVE_10Y_2Y",  # T10Y2Y
    "FED_FUNDS",           # DFF
]


def ingest_market_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    print(f"[*] Ingesting {len(TARGET_SYMBOLS)} instruments from Yahoo Finance [{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}]...")
    collector = YahooDataCollector()
    collected_bars = []

    for sym in TARGET_SYMBOLS:
        print(f"    -> Fetching {sym:8s} ...", end=" ", flush=True)
        try:
            df = collector.fetch_bars(sym, start_date=start_date, end_date=end_date)
            # Basic validation
            MarketDataValidator.assert_valid(df, sym)
            collected_bars.append(df)
            print(f"OK ({len(df):4d} bars)")
        except Exception as e:
            print(f"FAILED: {e}")

    if not collected_bars:
        raise RuntimeError("No market data could be fetched.")

    raw_bars = pd.concat(collected_bars, ignore_index=True)
    
    # Align multi-asset universe to SPY trading calendar
    print("[*] Harmonizing multi-asset timestamps to benchmark calendar...")
    calendar = MarketCalendar()
    aligned_bars = calendar.align_universe_bars(raw_bars, benchmark_symbol="SPY")
    print(f"    -> Harmonized dataset contains {len(aligned_bars)} rows across {len(aligned_bars['symbol'].unique())} assets.")
    return aligned_bars


def ingest_macro_data() -> pd.DataFrame:
    print(f"[*] Ingesting macroeconomic series from St. Louis FRED...")
    fred = FREDCollector()
    macro_records = []

    for series_name in FRED_MACRO_SERIES:
        print(f"    -> Fetching FRED series {series_name:20s} ...", end=" ", flush=True)
        try:
            df = fred.fetch_macro_series(series_name)
            macro_records.append(df)
            print(f"OK ({len(df):4d} observations)")
        except Exception as e:
            print(f"FAILED: {e}")

    if macro_records:
        return pd.concat(macro_records, ignore_index=True)
    return pd.DataFrame()


def main():
    print("=" * 70)
    print("[*] RAMP DATA INGESTION: REAL-WORLD OPEN-SOURCE LAKEHOUSE BUILDER")
    print("=" * 70)

    start_date = datetime(2018, 1, 1)
    end_date = datetime.now()

    # 1. Fetch & Harmonize Market Bars
    bars_df = ingest_market_data(start_date, end_date)

    # 2. Fetch FRED Macro Series
    macro_df = ingest_macro_data()

    # 3. Store Parquet Snapshots
    parquet_bars_file = PARQUET_DIR / "multi_asset_bars_2018_present.parquet"
    bars_df.to_parquet(parquet_bars_file, index=False)
    print(f"\n[*] Exported Parquet snapshot: {parquet_bars_file} ({parquet_bars_file.stat().st_size / 1e6:.2f} MB)")

    if not macro_df.empty:
        parquet_macro_file = PARQUET_DIR / "macro_indicators_fred.parquet"
        macro_df.to_parquet(parquet_macro_file, index=False)
        print(f"[*] Exported Macro Parquet: {parquet_macro_file}")

    # 4. Ingest into DuckDB Lakehouse
    print(f"\n[*] Populating persistent DuckDB lakehouse at {DUCKDB_PATH} ...")
    store = PointInTimeStore(DUCKDB_PATH)
    store.insert_bars(bars_df)
    if not macro_df.empty:
        store.insert_macro(macro_df)
    store.close()

    print("\n" + "=" * 70)
    print("[OK] Real-World Lakehouse Ingestion Complete!")
    print(f"  Total Trading Days Indexed : {len(bars_df['timestamp'].unique())}")
    print(f"  Asset Count                : {len(bars_df['symbol'].unique())}")
    print(f"  Date Span                  : {bars_df['timestamp'].min().strftime('%Y-%m-%d')} to {bars_df['timestamp'].max().strftime('%Y-%m-%d')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
