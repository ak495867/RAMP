import os
from pathlib import Path
import numpy as np
import pandas as pd

def generate_series(start_date="2020-01-01", end_date="2026-03-01", initial_price=100.0, mu=0.08, sigma=0.18):
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n = len(dates)
    dt = 1.0 / 252.0
    daily_returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    prices = initial_price * np.exp(np.cumsum(daily_returns))

    opens = prices * (1.0 + np.random.normal(0, 0.003, n))
    highs = np.maximum(prices, opens) * (1.0 + np.abs(np.random.normal(0, 0.006, n)))
    lows = np.minimum(prices, opens) * (1.0 - np.abs(np.random.normal(0, 0.006, n)))
    closes = prices
    volumes = np.random.lognormal(mean=14.0, sigma=0.5, size=n)

    return pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": np.round(volumes, 0)
    })

def main():
    np.random.seed(42)
    out_dir = Path("data/custom_csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    tech_df = generate_series(initial_price=150.0, mu=0.15, sigma=0.22)
    tech_df.to_csv(out_dir / "CUSTOM_TECH_INDEX.csv", index=False)

    vol_df = generate_series(initial_price=20.0, mu=0.0, sigma=0.45)
    vol_df_renamed = vol_df.rename(columns={
        "timestamp": "Date",
        "open": "PX_OPEN",
        "high": "PX_HIGH",
        "low": "PX_LOW",
        "close": "PX_CLOSE",
        "volume": "VOL"
    })
    vol_df_renamed.to_csv(out_dir / "CUSTOM_VOL_INDEX.csv", index=False, sep=";")

    comm_df = generate_series(initial_price=80.0, mu=0.06, sigma=0.25)
    comm_df_renamed = comm_df.rename(columns={
        "timestamp": "Datetime",
        "open": "Open Price",
        "high": "High Price",
        "low": "Low Price",
        "close": "Close Price",
        "volume": "Total Volume"
    })
    comm_df_renamed.to_csv(out_dir / "CUSTOM_COMMODITY.csv", index=False)

    macro_df = generate_series(initial_price=105.0, mu=0.03, sigma=0.08)
    macro_df.to_csv(out_dir / "CUSTOM_MACRO_FEED.csv", index=False)

    print(f"Generated 4 custom market datasets in {out_dir}")

if __name__ == "__main__":
    main()
