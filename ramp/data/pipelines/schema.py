from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class ColumnMapping:
    DEFAULT_TIMESTAMP_ALIASES = [
        "timestamp", "date", "datetime", "time", "date/time", "dt", "ts"
    ]
    DEFAULT_SYMBOL_ALIASES = [
        "symbol", "ticker", "asset", "instrument", "security"
    ]
    DEFAULT_OPEN_ALIASES = [
        "open", "px_open", "open_price", "first"
    ]
    DEFAULT_HIGH_ALIASES = [
        "high", "px_high", "high_price", "max"
    ]
    DEFAULT_LOW_ALIASES = [
        "low", "px_low", "low_price", "min"
    ]
    DEFAULT_CLOSE_ALIASES = [
        "close", "px_close", "close_price", "last", "adj_close", "adjusted_close"
    ]
    DEFAULT_VOLUME_ALIASES = [
        "volume", "vol", "shares", "total_volume", "size"
    ]

    def __init__(self, custom_mapping: Optional[Dict[str, str]] = None):
        self.custom_mapping = custom_mapping or {}

    def resolve_columns(self, available_columns: List[str]) -> Dict[str, str]:
        resolved = {}
        lookup = {col.lower().strip().replace(" ", "_"): col for col in available_columns}

        for target, aliases in [
            ("timestamp", self.DEFAULT_TIMESTAMP_ALIASES),
            ("symbol", self.DEFAULT_SYMBOL_ALIASES),
            ("open", self.DEFAULT_OPEN_ALIASES),
            ("high", self.DEFAULT_HIGH_ALIASES),
            ("low", self.DEFAULT_LOW_ALIASES),
            ("close", self.DEFAULT_CLOSE_ALIASES),
            ("volume", self.DEFAULT_VOLUME_ALIASES),
        ]:
            if target in self.custom_mapping:
                mapped = self.custom_mapping[target]
                if mapped in available_columns:
                    resolved[target] = mapped
                    continue

            found = False
            for alias in aliases:
                normalized_alias = alias.lower().strip().replace(" ", "_")
                if normalized_alias in lookup:
                    resolved[target] = lookup[normalized_alias]
                    found = True
                    break

        return resolved

class SchemaNormalizer:
    REQUIRED_CANONICAL_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, column_mapping: Optional[ColumnMapping] = None):
        self.column_mapping = column_mapping or ColumnMapping()

    def normalize(
        self,
        df: pd.DataFrame,
        default_symbol: Optional[str] = None
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

        resolved = self.column_mapping.resolve_columns(list(df.columns))

        missing = [col for col in self.REQUIRED_CANONICAL_COLUMNS if col not in resolved]
        if missing:
            raise ValueError(f"Unable to resolve required columns {missing} from input columns: {list(df.columns)}")

        inv_map = {orig: target for target, orig in resolved.items()}
        mapped_df = df.rename(columns=inv_map)

        if "symbol" not in mapped_df.columns:
            if not default_symbol:
                raise ValueError("Symbol column not present and default_symbol not provided")
            mapped_df["symbol"] = str(default_symbol).strip().upper()
        else:
            mapped_df["symbol"] = mapped_df["symbol"].astype(str).str.strip().str.upper()

        mapped_df["timestamp"] = pd.to_datetime(mapped_df["timestamp"], utc=True)
        mapped_df["timestamp"] = mapped_df["timestamp"].dt.tz_convert(None)

        for col in ["open", "high", "low", "close", "volume"]:
            mapped_df[col] = pd.to_numeric(mapped_df[col], errors="coerce")

        mapped_df = mapped_df.dropna(subset=["timestamp", "open", "high", "low", "close"])

        mapped_df["volume"] = mapped_df["volume"].fillna(0.0).clip(lower=0.0)

        mapped_df["high"] = np.maximum(
            mapped_df["high"],
            np.maximum(mapped_df["open"], mapped_df["close"])
        )
        mapped_df["low"] = np.minimum(
            mapped_df["low"],
            np.minimum(mapped_df["open"], mapped_df["close"])
        )

        canonical = mapped_df[
            ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        ].sort_values(by=["symbol", "timestamp"]).drop_duplicates(subset=["timestamp", "symbol"], keep="last")

        return canonical.reset_index(drop=True)
