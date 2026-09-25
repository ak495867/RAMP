# 7. Live Multi-Broker OMS/EMS & Cloud Operations

---

## 7.1 Automated Portfolio Drift Reconciler

In live production, physical positions drift away from target weights due to price movement, cash dividends, corporate actions, and partial fills. Continuous rebalancing creates unnecessary turnover and fee drag.

In [`ramp/paper/reconciler.py`](../ramp/paper/reconciler.py), RAMP enforces an **Automated Drift Reconciler**:

$$\text{Drift}_i = \left| \frac{\text{Position Market Value}_i}{\text{Total Account NAV}} - w_i^* \right|$$

* **Tolerance Band ($\tau_{\text{drift}} = 1.0\%$)**: If drift is below $1.0\%$, trading is suppressed.
* **Minimum Dollar Threshold**: Trades below $\$100$ are filtered out to prevent micro-order fee penalties.
* **Order Generation**: Generates clean target rebalancing orders only when portfolio drift breaches tolerance limits.

---

## 7.2 Broker Connectivity: Alpaca & IBKR Gateways

RAMP supports unified broker interfaces inheriting from [`BaseBroker`](../ramp/paper/simulated_broker.py#L9-L22):

### 1. Alpaca Securities Gateway
* **File**: [`ramp/paper/alpaca_broker.py`](../ramp/paper/alpaca_broker.py)
* **Features**: Zero-commission US equity and ETF execution via REST API / WebSockets. Supports paper trading and live account modes.

### 2. High-Fidelity Simulated Broker
* **File**: [`ramp/paper/simulated_broker.py`](../ramp/paper/simulated_broker.py)
* **Features**: Local in-memory paper broker tracking position cost basis, cash accrual, and order execution logs.

---

## 7.3 Multi-Channel Webhook Alerting Service

In [`ramp/core/alerts.py`](../ramp/core/alerts.py), RAMP provides automated real-time alerts dispatched to **Discord** and **Slack**:

### Supported Critical Events:
1. **Circuit-Breaker Kill-Switch Triggered**: Dispatches an emergency payload with current drawdown percentage and liquidation status.
2. **Regime Transition Detected**: Notifies the risk team of macro state changes (e.g. *Low-Vol Expansion $\to$ Crisis Shock*).
3. **Daily Rebalance Executed**: Summarizes executed child orders, total volume, and realized frictional slippage.
4. **Pre-Trade Compliance Rejections**: Dispatches warnings if ADV caps or single-name concentration limits are violated.

---

## 7.4 Cloud Containerization: Docker Compose Architecture

RAMP is packaged into a production **Docker Compose** orchestration:

```yaml
version: '3.8'

services:
  ramp-api:
    build: .
    container_name: ramp-api-service
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - ENVIRONMENT=production
      - DUCKDB_PATH=/app/data/ramp_lakehouse.duckdb
    restart: unless-stopped

  ramp-dashboard:
    build: .
    container_name: ramp-dashboard-ui
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    command: ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    depends_on:
      - ramp-api
    restart: unless-stopped

  ramp-redis:
    image: redis:7-alpine
    container_name: ramp-redis-cache
    ports:
      - "6379:6379"
    restart: unless-stopped
```

### Production Launch Commands:

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f ramp-api
```

* **FastAPI Service**: Available at `http://localhost:8000`
* **Streamlit Research Terminal**: Available at `http://localhost:8501`
* **Redis Cache**: Available on port `6379`
