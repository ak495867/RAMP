import os
from pathlib import Path
import pandas as pd
from ramp.data.pit_store import PointInTimeStore
from ramp.data.pipelines.csv_pipeline import CSVPipeline
from ramp.data.pipelines.parquet_pipeline import ParquetPipeline
from ramp.data.pipelines.manager import PipelineManager

def main():
    lakehouse_path = Path("data/ramp_lakehouse.duckdb")
    store = PointInTimeStore(lakehouse_path)
    manager = PipelineManager(store=store)

    csv_dir = Path("data/custom_csv")
    csv_results = manager.ingest_csv_directory(csv_dir)
    print(f"Ingested CSVs from {csv_dir}: {csv_results['rows']} bars across {csv_results.get('symbols', [])}")

    parquet_dir = Path("data/parquet/custom_assets")
    parquet_pipeline = ParquetPipeline()

    clean_bars = store.get_bars_pit(
        symbols=csv_results.get("symbols", []),
        start_date=pd.to_datetime("2020-01-01"),
        end_date=pd.to_datetime("2026-03-01")
    )

    if not clean_bars.empty:
        parquet_pipeline.write_dataset(
            df=clean_bars,
            output_dir=parquet_dir,
            partition_by=["symbol"]
        )
        print(f"Exported partitioned Parquet dataset to {parquet_dir}")

    meta = parquet_pipeline.get_metadata(parquet_dir)
    print("Partitioned Parquet Metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    summary = manager.get_summary()
    print("DuckDB Lakehouse Summary:")
    print(f"  Total market bars: {summary['market_bars']['total_bars']}")
    print(f"  Unique symbols: {summary['market_bars']['unique_symbols']}")
    print(f"  Date range: {summary['market_bars']['min_bar_date']} to {summary['market_bars']['max_bar_date']}")

    store.close()

if __name__ == "__main__":
    main()
