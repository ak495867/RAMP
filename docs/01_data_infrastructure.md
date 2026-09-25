# 1. Data Infrastructure & Point-in-Time Lakehouse

---

## 1.1 The Look-Ahead Leakage Trap

In quantitative research, data leakage occurs when future information is inadvertently incorporated into historical decisions. Common failure modes include:
* Using dividend-adjusted price series calculated with hindsight dividends.
* Stitching futures contracts with unadjusted roll gaps that create artificial return shocks.
* Calculating rolling indicators on day $t$ using day $t$ close to simulate fills at day $t$ close.
* Misaligning economic indicator releases (e.g. GDP or CPI) that were revised months after their preliminary publication date.

---

## 1.2 Point-in-Time DuckDB Lakehouse Architecture

RAMP implements an embedded columnar OLAP store using **DuckDB** backed by partitioned **Apache Parquet** snapshots:

```
d:/RAMP/data/
├── parquet/
│   ├── multi_asset_bars_2018_present.parquet
│   └── macro_indicators_fred.parquet
└── ramp_lakehouse.duckdb
```

### Table Schemas

```sql
CREATE TABLE market_bars (
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    PRIMARY KEY (timestamp, symbol)
);

CREATE TABLE macro_indicators (
    timestamp TIMESTAMP NOT NULL,
    series_name VARCHAR NOT NULL,
    value DOUBLE NOT NULL,
    PRIMARY KEY (timestamp, series_name)
);
```

### Zero Look-Ahead SQL Window Formulation

When generating rolling indicators (returns, volatility, correlation) for day $T$, RAMP enforces point-in-time window functions:

```sql
SELECT 
    timestamp,
    symbol,
    close,
    (close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp)) / 
        NULLIF(LAG(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp), 0) AS daily_ret
FROM market_bars
WHERE timestamp <= ?
```

---

## 1.3 Continuous Futures Roll Stitching

Futures contracts expire on fixed delivery schedules. To construct continuous price series without artificial price gaps, RAMP supports two methodologies in [`ramp/data/rolls.py`](file:///d:/RAMP/ramp/data/rolls.py):

### 1. Ratio Backward Adjustment (Percentage Preserving)
Multiplies historical prices by the ratio of the next contract to the front contract on the roll date:
$$P_t^{\text{adj}} = P_t \times \prod_{k \in \text{Rolls}, t \le t_k} \frac{P_{t_k}^{\text{next}}}{P_{t_k}^{\text{front}}}$$
*Best for: Financial index futures (ES, NQ) and currencies, preserving percentage returns.*

### 2. Panama Canal Adjustment (Spread Preserving)
Adds the cumulative price gap between the next and front contract backward across history:
$$P_t^{\text{adj}} = P_t + \sum_{k \in \text{Rolls}, t \le t_k} \left(P_{t_k}^{\text{next}} - P_{t_k}^{\text{front}}\right)$$
*Best for: Physical commodities (CL, GC) and calendar spread modeling.*

### Roll Yield & Basis Calculation
Annualized roll yield measures the structural carry return from the term structure:
$$\text{Roll Yield} = \frac{F_{\text{front}} - F_{\text{next}}}{F_{\text{front}}} \times \frac{365}{\text{Days to Expiry}}$$
* Positive = Backwardation (downward sloping curve, positive roll return for longs).
* Negative = Contango (upward sloping curve, negative roll drag for longs).

---

## 1.4 Multi-Asset Calendar Harmonization

Traditional equity markets operate Monday through Friday (09:30 - 16:00 EST) with bank holidays, whereas digital assets trade 24/7/365. 

In [`ramp/core/calendar.py`](file:///d:/RAMP/ramp/core/calendar.py), RAMP harmonizes all instruments to the benchmark (`SPY`) active trading calendar:
1. Weekend-only crypto bars are rolled into Monday trading bars.
2. Missing prints on foreign exchange or sovereign bond holidays are forward-filled using the previous closing price with zero volume to preserve point-in-time correctness without look-ahead interpolation.
