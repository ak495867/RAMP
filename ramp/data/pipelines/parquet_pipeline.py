from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import duckdb
import pandas as pd
from ramp.data.pit_store import PointInTimeStore

class ParquetPipeline:
    def __init__(self, default_compression: str = "snappy"):
        self.default_compression = default_compression

    def write_dataset(
        self,
        df: pd.DataFrame,
        output_dir: Union[str, Path],
        partition_by: Optional[List[str]] = None
    ) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        save_df = df.copy()
        if "year" in (partition_by or []) and "year" not in save_df.columns:
            save_df["year"] = pd.to_datetime(save_df["timestamp"]).dt.year

        save_df.to_parquet(
            out,
            index=False,
            partition_cols=partition_by,
            compression=self.default_compression
        )
        return out

    def read_dataset(
        self,
        path: Union[str, Path],
        symbols: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> pd.DataFrame:
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Parquet source not found: {target_path}")

        path_str = str(target_path).replace("\\", "/")
        if target_path.is_dir():
            query_source = f"'{path_str}/**/*.parquet'"
        else:
            query_source = f"'{path_str}'"

        select_cols = ", ".join(columns) if columns else "*"
        clauses = []

        if symbols:
            sym_list = ", ".join([f"'{s}'" for s in symbols])
            clauses.append(f"symbol IN ({sym_list})")

        if start_date is not None:
            clauses.append(f"timestamp >= '{pd.to_datetime(start_date).isoformat()}'")

        if end_date is not None:
            clauses.append(f"timestamp <= '{pd.to_datetime(end_date).isoformat()}'")

        where_stmt = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT {select_cols} FROM read_parquet({query_source}) {where_stmt} ORDER BY timestamp ASC, symbol ASC"

        conn = duckdb.connect(":memory:")
        try:
            df = conn.execute(query).fetchdf()
            return df
        finally:
            conn.close()

    def get_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
        target_path = Path(path)
        path_str = str(target_path).replace("\\", "/")
        query_source = f"'{path_str}/**/*.parquet'" if target_path.is_dir() else f"'{path_str}'"

        conn = duckdb.connect(":memory:")
        try:
            stats = conn.execute(f"""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(timestamp) as min_timestamp,
                    MAX(timestamp) as max_timestamp
                FROM read_parquet({query_source})
            """).fetchdf().iloc[0].to_dict()

            symbols = conn.execute(f"""
                SELECT DISTINCT symbol FROM read_parquet({query_source}) ORDER BY symbol ASC
            """).fetchdf()["symbol"].tolist()

            stats["symbols"] = symbols
            return stats
        finally:
            conn.close()

    def ingest_into_lakehouse(
        self,
        path: Union[str, Path],
        store: PointInTimeStore,
        symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> int:
        df = self.read_dataset(
            path=path,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        if df.empty:
            return 0

        store.insert_bars(df)
        return len(df)
