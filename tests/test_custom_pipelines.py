from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from ramp.data.pit_store import PointInTimeStore
from ramp.data.pipelines.schema import ColumnMapping, SchemaNormalizer
from ramp.data.pipelines.csv_pipeline import CSVPipeline
from ramp.data.pipelines.parquet_pipeline import ParquetPipeline
from ramp.data.pipelines.manager import PipelineManager

def test_schema_normalizer_column_aliases():
    df = pd.DataFrame({
        "Date": ["2023-01-03", "2023-01-04"],
        "PX_OPEN": [100.0, 102.0],
        "PX_HIGH": [105.0, 106.0],
        "PX_LOW": [99.0, 101.0],
        "PX_CLOSE": [103.0, 104.0],
        "VOL": [50000, 60000]
    })
    normalizer = SchemaNormalizer()
    clean = normalizer.normalize(df, default_symbol="TEST_ASSET")

    assert list(clean.columns) == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    assert clean["symbol"].iloc[0] == "TEST_ASSET"
    assert clean["close"].iloc[1] == 104.0
    assert len(clean) == 2

def test_schema_normalizer_invariant_fixing():
    df = pd.DataFrame({
        "timestamp": ["2023-01-03"],
        "symbol": ["XYZ"],
        "open": [100.0],
        "high": [95.0],
        "low": [105.0],
        "close": [102.0],
        "volume": [-500.0]
    })
    normalizer = SchemaNormalizer()
    clean = normalizer.normalize(df)

    assert clean["high"].iloc[0] >= 102.0
    assert clean["low"].iloc[0] <= 100.0
    assert clean["volume"].iloc[0] == 0.0

def test_csv_pipeline_delimiter_detection_and_ingest():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_file = tmp_path / "semicolon_asset.csv"

        raw_csv = (
            "date;open;high;low;close;volume\n"
            "2023-05-01;50.0;52.0;49.0;51.5;10000\n"
            "2023-05-02;51.5;53.0;51.0;52.8;12000\n"
        )
        csv_file.write_text(raw_csv)

        pipeline = CSVPipeline()
        detected = pipeline.detect_delimiter(csv_file)
        assert detected == ";"

        df = pipeline.read_file(csv_file, symbol_override="SEMI")
        assert len(df) == 2
        assert df["symbol"].iloc[0] == "SEMI"
        assert df["close"].iloc[1] == 52.8

        bars = pipeline.to_bars(df)
        assert len(bars) == 2
        assert bars[0].symbol == "SEMI"

def test_parquet_pipeline_partitioned_read_write():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        pq_dir = tmp_path / "parquet_store"

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"]),
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "open": [10.0, 11.0, 20.0, 21.0],
            "high": [12.0, 13.0, 22.0, 23.0],
            "low": [9.0, 10.0, 19.0, 20.0],
            "close": [11.5, 12.5, 21.5, 22.5],
            "volume": [1000.0, 1500.0, 2000.0, 2500.0]
        })

        pq_pipe = ParquetPipeline()
        pq_pipe.write_dataset(df, pq_dir, partition_by=["symbol"])

        meta = pq_pipe.get_metadata(pq_dir)
        assert meta["total_rows"] == 4
        assert meta["unique_symbols"] == 2
        assert sorted(meta["symbols"]) == ["AAA", "BBB"]

        filtered_df = pq_pipe.read_dataset(pq_dir, symbols=["AAA"])
        assert len(filtered_df) == 2
        assert (filtered_df["symbol"] == "AAA").all()

def test_pipeline_manager_end_to_end_lakehouse():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_dir = tmp_path / "feeds"
        csv_dir.mkdir()

        f1 = csv_dir / "ASSET1.csv"
        f1.write_text("timestamp,open,high,low,close,volume\n2023-01-01,10,12,9,11,1000\n2023-01-02,11,13,10,12,1200\n")

        f2 = csv_dir / "ASSET2.csv"
        f2.write_text("timestamp,open,high,low,close,volume\n2023-01-01,50,55,48,53,500\n2023-01-02,53,56,51,54,600\n")

        store = PointInTimeStore(":memory:")
        manager = PipelineManager(store=store)

        res = manager.ingest_csv_directory(csv_dir)
        assert res["status"] == "success"
        assert res["rows"] == 4
        assert sorted(res["symbols"]) == ["ASSET1", "ASSET2"]

        summary = manager.get_summary()
        assert summary["market_bars"]["total_bars"] == 4
        assert summary["market_bars"]["unique_symbols"] == 2

        pit_bars = store.get_bars_pit(
            symbols=["ASSET1"],
            start_date=pd.to_datetime("2023-01-01"),
            end_date=pd.to_datetime("2023-01-02")
        )
        assert len(pit_bars) == 2
        store.close()
