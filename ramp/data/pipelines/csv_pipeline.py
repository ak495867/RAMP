import csv
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from ramp.core.types import Bar
from ramp.data.pit_store import PointInTimeStore
from ramp.data.pipelines.schema import ColumnMapping, SchemaNormalizer

class CSVPipeline:
    def __init__(
        self,
        column_mapping: Optional[ColumnMapping] = None,
        default_delimiter: Optional[str] = None
    ):
        self.normalizer = SchemaNormalizer(column_mapping)
        self.default_delimiter = default_delimiter

    def detect_delimiter(self, file_path: Union[str, Path]) -> str:
        if self.default_delimiter:
            return self.default_delimiter

        path = Path(file_path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)

        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except Exception:
            if ";" in sample and "," not in sample:
                return ";"
            if "\t" in sample and "," not in sample:
                return "\t"
            if "|" in sample and "," not in sample:
                return "|"
            return ","

    def read_file(
        self,
        file_path: Union[str, Path],
        symbol_override: Optional[str] = None
    ) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        delimiter = self.detect_delimiter(path)
        raw_df = pd.read_csv(path, sep=delimiter)

        sym = symbol_override or path.stem.upper()
        clean_df = self.normalizer.normalize(raw_df, default_symbol=sym)
        return clean_df

    def read_directory(
        self,
        dir_path: Union[str, Path],
        file_glob: str = "*.csv",
        recursive: bool = False
    ) -> pd.DataFrame:
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {path}")

        pattern = f"**/{file_glob}" if recursive else file_glob
        csv_files = sorted(list(path.glob(pattern)))

        frames = []
        for file in csv_files:
            try:
                frame = self.read_file(file)
                if not frame.empty:
                    frames.append(frame)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(by=["symbol", "timestamp"]).drop_duplicates(
            subset=["timestamp", "symbol"], keep="last"
        )
        return combined.reset_index(drop=True)

    def to_bars(self, df: pd.DataFrame) -> List[Bar]:
        bars = []
        for row in df.itertuples(index=False):
            bars.append(
                Bar(
                    timestamp=row.timestamp,
                    symbol=row.symbol,
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                )
            )
        return bars

    def export_to_parquet(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        compression: str = "snappy"
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression=compression)
        return out

    def ingest_into_lakehouse(
        self,
        source: Union[str, Path, pd.DataFrame],
        store: PointInTimeStore,
        symbol_override: Optional[str] = None
    ) -> int:
        if isinstance(source, pd.DataFrame):
            clean_df = self.normalizer.normalize(source, default_symbol=symbol_override)
        else:
            path = Path(source)
            if path.is_dir():
                clean_df = self.read_directory(path)
            else:
                clean_df = self.read_file(path, symbol_override=symbol_override)

        if clean_df.empty:
            return 0

        store.insert_bars(clean_df)
        return len(clean_df)
