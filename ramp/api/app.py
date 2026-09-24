"""
FastAPI Production Microservice for RAMP.
Exposes endpoints for real-time regime inference, portfolio telemetry, and backtest execution.
"""

from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from ramp.data.collectors.synthetic import SyntheticRegimeDataGenerator
from ramp.regimes.online_hmm import OnlineHamiltonFilterHMM
from ramp.regimes.filter import RegimeHysteresisFilter
from ramp.signals.momentum import TimeSeriesMomentumSignal
from ramp.signals.carry import CrossAssetCarrySignal
from ramp.backtest.engine import EventDrivenBacktestEngine
from ramp.paper.simulated_broker import SimulatedPaperBroker

app = FastAPI(
    title="RAMP: Quantitative Portfolio & Regime API",
    description="Research-to-Production Regime-Aware Multi-Asset Execution Engine",
    version="0.1.0"
)

# Global in-memory state
broker = SimulatedPaperBroker(initial_cash=1000000.0)


class BacktestRequest(BaseModel):
    symbols: List[str] = ["SPY", "TLT", "GLD", "BTC-USD"]
    n_bars: int = 250
    rebalance_freq: int = 5
    initial_capital: float = 1000000.0


@app.get("/")
def root():
    return {
        "system": "RAMP: Regime-Adaptive Multi-Asset Platform",
        "status": "online",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "service": "ramp-engine", "uptime": "ok"}


@app.get("/api/v1/broker/portfolio")
def get_portfolio():
    nav = broker.get_nav()
    positions = broker.get_positions()
    pos_data = {
        sym: {"qty": p.quantity, "avg_price": p.avg_price, "market_value": p.market_value}
        for sym, p in positions.items()
    }
    return {
        "nav": nav,
        "cash": broker.cash,
        "positions": pos_data
    }


@app.post("/api/v1/backtest/run")
def run_backtest_endpoint(req: BacktestRequest):
    try:
        gen = SyntheticRegimeDataGenerator(seed=42)
        bars, _ = gen.generate_universe(req.symbols, n_bars=req.n_bars)

        hmm = OnlineHamiltonFilterHMM(n_regimes=3)
        hyst = RegimeHysteresisFilter()
        signals = [
            TimeSeriesMomentumSignal(lookbacks=[21, 63]),
            CrossAssetCarrySignal()
        ]

        engine = EventDrivenBacktestEngine(
            symbols=req.symbols,
            regime_detector=hmm,
            hysteresis_filter=hyst,
            signals=signals,
            initial_capital=req.initial_capital,
            rebalance_frequency_bars=req.rebalance_freq,
            burn_in_bars=50
        )

        results = engine.run(bars)
        return {
            "metrics": results["metrics"],
            "total_trades": len(results["fills"]),
            "final_nav": float(results["history"]["total_nav"].iloc[-1]) if not results["history"].empty else req.initial_capital
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
