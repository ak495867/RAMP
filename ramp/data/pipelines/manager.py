from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from ramp.data.pit_store import PointInTimeStore
from ramp.data.validator import MarketDataValidator
from ramp.data.rolls import ContinuousFuturesBuilder
from ramp.data.pipelines.csv_pipeline import CSVPipeline
from ramp.data.pipelines.parquet_pipeline import ParquetPipeline
from ramp.data.pipelines.schema import ColumnMapping

class PipelineManager:
    def __init__(
        self,
        store: Optional[PointInTimeStore] = None,
        column_mapping: Optional[ColumnMapping] = None
    ):
        self.store = store or PointInTimeStore(":memory:")
        self.csv_pipeline = CSVPipeline(column_mapping=column_mapping)
        self.parquet_pipeline = ParquetPipeline()
        self.validator = MarketDataValidator()
        self.futures_builder = ContinuousFuturesBuilder()

    def ingest_csv_file(
        self,
        file_path: Union[str, Path],
        symbol_override: Optional[str] = None
    ) -> Dict[str, Any]:
        df = self.csv_pipeline.read_file(file_path, symbol_override=symbol_override)
        if df.empty:
            return {"status": "empty", "rows": 0, "symbol": symbol_override}

        sym = df["symbol"].iloc[0]
        self.validator.assert_valid(df, sym)
        self.store.insert_bars(df)
        return {
            "status": "success",
            "source": str(file_path),
            "symbol": sym,
            "rows": len(df),
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max())
        }

    def ingest_csv_directory(
        self,
        dir_path: Union[str, Path],
        file_glob: str = "*.csv",
        recursive: bool = False
    ) -> Dict[str, Any]:
        df = self.csv_pipeline.read_directory(dir_path, file_glob=file_glob, recursive=recursive)
        if df.empty:
            return {"status": "empty", "rows": 0, "symbols": []}

        for sym, sym_df in df.groupby("symbol"):
            self.validator.assert_valid(sym_df, str(sym))

        self.store.insert_bars(df)
        symbols = sorted(df["symbol"].unique().tolist())
        return {
            "status": "success",
            "source_dir": str(dir_path),
            "symbols": symbols,
            "rows": len(df),
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max())
        }

    def ingest_parquet(
        self,
        path: Union[str, Path],
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        count = self.parquet_pipeline.ingest_into_lakehouse(
            path=path,
            store=self.store,
            symbols=symbols
        )
        return {
            "status": "success",
            "source": str(path),
            "rows": count
        }

    def export_lakehouse_to_parquet(
        self,
        symbols: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        output_dir: Union[str, Path],
        partition_by: Optional[List[str]] = None
    ) -> Path:
        df = self.store.get_bars_pit(symbols=symbols, start_date=start_date, end_date=end_date)
        if df.empty:
            raise ValueError(f"No bars found for symbols {symbols} between {start_date} and {end_date}")

        return self.parquet_pipeline.write_dataset(
            df=df,
            output_dir=output_dir,
            partition_by=partition_by
        )

    def get_summary(self) -> Dict[str, Any]:
        res = self.store.conn.execute("""
            SELECT 
                COUNT(*) as total_bars,
                COUNT(DISTINCT symbol) as unique_symbols,
                MIN(timestamp) as min_bar_date,
                MAX(timestamp) as max_bar_date
            FROM market_bars
        """).fetchdf().iloc[0].to_dict()

        macro_res = self.store.conn.execute("""
            SELECT 
                COUNT(*) as total_macro,
                COUNT(DISTINCT series_name) as unique_series
            FROM macro_indicators
        """).fetchdf().iloc[0].to_dict()

        return {
            "market_bars": res,
            "macro_indicators": macro_res
        }
