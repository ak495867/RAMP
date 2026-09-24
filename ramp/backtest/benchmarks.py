"""
Classic Benchmark Portfolios for Quantitative Comparison.
Implements:
1. Static 60/40 Equity/Bond Benchmark (60% SPY / 40% TLT)
2. Static Risk Parity / Inverse Volatility (1 / sigma_i)
3. Equal Weight (1 / N)
"""

from typing import Dict, List
import numpy as np
import pandas as pd
from ramp.execution.accounting import PortfolioLedger
from ramp.execution.cost_model import ExecutionCostModel
from ramp.core.types import Fill, Order, OrderSide, OrderType
from ramp.backtest.metrics import PerformanceMetricsCalculator


class BenchmarkEvaluator:
    """
    Simulates benchmark allocations under the same non-linear transaction cost and latency model.
    """

    @staticmethod
    def run_static_60_40(
        bars_df: pd.DataFrame,
        equity_sym: str = "SPY",
        bond_sym: str = "TLT",
        rebalance_freq_bars: int = 21,
        initial_capital: float = 1000000.0,
        cost_model: Optional[ExecutionCostModel] = None
    ) -> Dict:
        """Runs 60% Equity / 40% Long Treasury benchmark."""
        cost = cost_model or ExecutionCostModel()
        ledger = PortfolioLedger(initial_capital=initial_capital)

        df = bars_df.copy().sort_values(["timestamp", "symbol"])
        dates = sorted(df["timestamp"].unique())

        pending_orders: List[Order] = []
        turnover_acc = 0.0
        current_target_w = np.array([0.0, 0.0])  # [SPY, TLT]

        for i, dt in enumerate(dates):
            current_slice = df[df["timestamp"] == dt]
            prices = dict(zip(current_slice["symbol"], current_slice["close"]))
            opens = dict(zip(current_slice["symbol"], current_slice["open"]))
            vols = dict(zip(current_slice["symbol"], current_slice["volume"]))

            # Execute pending orders at today's OPEN
            for order in pending_orders:
                sym = order.symbol
                open_p = opens.get(sym, prices.get(sym, 100.0))
                v = vols.get(sym, 1000000.0)
                exec_qty, fill_p, comm, slip = cost.calculate_fill(
                    order.side, order.quantity, open_p, v, 0.015
                )
                if exec_qty > 0:
                    fill = Fill(
                        fill_id=f"bm_fill_{len(ledger.fills_history) + 1}",
                        order_id=order.order_id,
                        symbol=sym,
                        side=order.side,
                        quantity=exec_qty,
                        price=fill_p,
                        commission=comm,
                        slippage=slip,
                        timestamp=dt
                    )
                    ledger.record_fill(fill)
            pending_orders = []

            ledger.accrue_daily_interest(dt)
            nav_record = ledger.mark_to_market(dt, prices)
            nav = nav_record["total_nav"]

            # Rebalance
            if i % rebalance_freq_bars == 0 and i < len(dates) - 1:
                target_w = np.array([0.60, 0.40])
                turnover_acc += float(np.sum(np.abs(target_w - current_target_w)))
                current_target_w = target_w

                for s_idx, sym in enumerate([equity_sym, bond_sym]):
                    target_dollar = nav * target_w[s_idx]
                    p = prices.get(sym, 100.0)
                    target_shares = target_dollar / max(p, 1e-4)
                    curr_shares = ledger.positions[sym].quantity if sym in ledger.positions else 0.0
                    delta = target_shares - curr_shares
                    if abs(delta) >= 1.0:
                        pending_orders.append(Order(
                            order_id=f"bm_ord_{i}_{sym}",
                            symbol=sym,
                            side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                            quantity=abs(delta),
                            created_at=dt
                        ))

        hist = ledger.get_history_df()
        years = len(dates) / 252.0
        metrics = PerformanceMetricsCalculator.calculate_all(
            hist["daily_return"],
            turnover_annual=turnover_acc / max(years, 0.1)
        )
        return {"name": "Static 60/40 Benchmark", "metrics": metrics, "history": hist, "fills": ledger.fills_history}

    @staticmethod
    def run_static_risk_parity(
        bars_df: pd.DataFrame,
        symbols: List[str],
        rebalance_freq_bars: int = 21,
        lookback_vol: int = 63,
        initial_capital: float = 1000000.0,
        cost_model: Optional[ExecutionCostModel] = None
    ) -> Dict:
        """Runs Inverse-Volatility Risk Parity benchmark across all symbols."""
        cost = cost_model or ExecutionCostModel()
        ledger = PortfolioLedger(initial_capital=initial_capital)

        df = bars_df.copy().sort_values(["timestamp", "symbol"])
        dates = sorted(df["timestamp"].unique())
        N = len(symbols)

        pending_orders: List[Order] = []
        turnover_acc = 0.0
        current_target_w = np.zeros(N)

        for i, dt in enumerate(dates):
            current_slice = df[df["timestamp"] == dt]
            prices = dict(zip(current_slice["symbol"], current_slice["close"]))
            opens = dict(zip(current_slice["symbol"], current_slice["open"]))
            vols = dict(zip(current_slice["symbol"], current_slice["volume"]))

            for order in pending_orders:
                sym = order.symbol
                open_p = opens.get(sym, prices.get(sym, 100.0))
                v = vols.get(sym, 1000000.0)
                exec_qty, fill_p, comm, slip = cost.calculate_fill(
                    order.side, order.quantity, open_p, v, 0.015
                )
                if exec_qty > 0:
                    fill = Fill(
                        fill_id=f"rp_fill_{len(ledger.fills_history) + 1}",
                        order_id=order.order_id,
                        symbol=sym,
                        side=order.side,
                        quantity=exec_qty,
                        price=fill_p,
                        commission=comm,
                        slippage=slip,
                        timestamp=dt
                    )
                    ledger.record_fill(fill)
            pending_orders = []

            ledger.accrue_daily_interest(dt)
            nav_record = ledger.mark_to_market(dt, prices)
            nav = nav_record["total_nav"]

            # Compute inverse volatility weights
            if i % rebalance_freq_bars == 0 and i >= lookback_vol and i < len(dates) - 1:
                hist_window = df[df["timestamp"] <= dt]
                piv = hist_window.pivot(index="timestamp", columns="symbol", values="close")
                
                inv_vols = []
                for s in symbols:
                    if s in piv.columns and len(piv[s].dropna()) >= lookback_vol:
                        rets = piv[s].pct_change().dropna().values[-lookback_vol:]
                        std_s = float(np.std(rets, ddof=1))
                        inv_vols.append(1.0 / max(std_s, 1e-4))
                    else:
                        inv_vols.append(1.0)

                target_w = np.array(inv_vols) / sum(inv_vols)
                turnover_acc += float(np.sum(np.abs(target_w - current_target_w)))
                current_target_w = target_w

                for s_idx, sym in enumerate(symbols):
                    target_dollar = nav * target_w[s_idx]
                    p = prices.get(sym, 100.0)
                    target_shares = target_dollar / max(p, 1e-4)
                    curr_shares = ledger.positions[sym].quantity if sym in ledger.positions else 0.0
                    delta = target_shares - curr_shares
                    if abs(delta) >= 1.0:
                        pending_orders.append(Order(
                            order_id=f"rp_ord_{i}_{sym}",
                            symbol=sym,
                            side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                            quantity=abs(delta),
                            created_at=dt
                        ))

        hist = ledger.get_history_df()
        years = len(dates) / 252.0
        metrics = PerformanceMetricsCalculator.calculate_all(
            hist["daily_return"],
            turnover_annual=turnover_acc / max(years, 0.1)
        )
        return {"name": "Static Risk Parity Benchmark", "metrics": metrics, "history": hist, "fills": ledger.fills_history}
