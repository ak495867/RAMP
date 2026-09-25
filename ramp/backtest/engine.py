"""
Strictly Event-Driven Backtesting Engine.
Executes step-by-step simulation with zero look-ahead bias,
market impact slippage, delayed execution (T -> T+1 Open), and regime-conditioned allocation.
"""

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ramp.core.types import Fill, Order, OrderSide, OrderType, RegimeState
from ramp.execution.accounting import PortfolioLedger
from ramp.execution.cost_model import ExecutionCostModel
from ramp.portfolio.black_litterman import RegimeConditionedBlackLitterman
from ramp.portfolio.optimizer import RobustConvexOptimizer
from ramp.portfolio.vol_target import VolatilityTargetingEngine
from ramp.regimes.base import BaseRegimeDetector
from ramp.regimes.filter import RegimeHysteresisFilter
from ramp.signals.base import BaseSignal

class EventDrivenBacktestEngine:
    """
    Simulates the research-to-execution pipeline under realistic market frictions.
    """

    def __init__(
        self,
        symbols: List[str],
        regime_detector: BaseRegimeDetector,
        hysteresis_filter: RegimeHysteresisFilter,
        signals: List[BaseSignal],
        optimizer: Optional[RobustConvexOptimizer] = None,
        vol_targeting: Optional[VolatilityTargetingEngine] = None,
        cost_model: Optional[ExecutionCostModel] = None,
        initial_capital: float = 1000000.0,
        rebalance_frequency_bars: int = 5,                            
        burn_in_bars: int = 60
    ):
        self.symbols = symbols
        self.regime_detector = regime_detector
        self.hysteresis = hysteresis_filter
        self.signals = signals
        self.optimizer = optimizer or RobustConvexOptimizer()
        self.vol_targeting = vol_targeting or VolatilityTargetingEngine()
        self.cost_model = cost_model or ExecutionCostModel()
        self.bl = RegimeConditionedBlackLitterman()
        self.ledger = PortfolioLedger(initial_capital=initial_capital)
        self.rebalance_frequency = rebalance_frequency_bars
        self.burn_in_bars = burn_in_bars

        self.weights_history: List[Dict] = []
        self.regimes_history: List[Dict] = []
        self.turnover_accumulator: float = 0.0

    def run(self, bars_df: pd.DataFrame) -> Dict:
        """
        Executes the historical backtest loop.
        """
        df = bars_df.copy().sort_values(["timestamp", "symbol"])
        unique_dates = sorted(df["timestamp"].unique())

        if len(unique_dates) <= self.burn_in_bars:
            raise ValueError(f"Need at least {self.burn_in_bars + 10} dates, got {len(unique_dates)}")

        burn_in_dates = unique_dates[:self.burn_in_bars]
        burn_in_df = df[df["timestamp"].isin(burn_in_dates)]
        pivoted_burn_in = burn_in_df.pivot(index="timestamp", columns="symbol", values="close").pct_change().dropna()

        feature_burn_in = pd.DataFrame({
            "ret": pivoted_burn_in.mean(axis=1),
            "vol": pivoted_burn_in.std(axis=1) * np.sqrt(252)
        })
        self.regime_detector.fit(feature_burn_in)

        oos_dates = unique_dates[self.burn_in_bars:]
        current_target_weights = np.zeros(len(self.symbols))
        pending_orders: List[Order] = []

        for i, current_dt in enumerate(oos_dates):
            current_bar_slice = df[df["timestamp"] == current_dt]
            current_prices = dict(zip(current_bar_slice["symbol"], current_bar_slice["close"]))
            current_opens = dict(zip(current_bar_slice["symbol"], current_bar_slice["open"]))
            current_volumes = dict(zip(current_bar_slice["symbol"], current_bar_slice["volume"]))

            if pending_orders:
                for order in pending_orders:
                    sym = order.symbol
                    open_p = current_opens.get(sym, current_prices.get(sym, 100.0))
                    vol = current_volumes.get(sym, 1000000.0)

                    exec_qty, fill_p, comm, slip = self.cost_model.calculate_fill(
                        side=order.side,
                        requested_qty=order.quantity,
                        mid_price=open_p,
                        daily_volume=vol,
                        daily_volatility=0.015
                    )

                    if exec_qty > 0:
                        fill = Fill(
                            fill_id=f"fill_{len(self.ledger.fills_history) + 1}",
                            order_id=order.order_id,
                            symbol=sym,
                            side=order.side,
                            quantity=exec_qty,
                            price=fill_p,
                            commission=comm,
                            slippage=slip,
                            timestamp=current_dt
                        )
                        self.ledger.record_fill(fill)

                pending_orders = []

            self.ledger.accrue_daily_interest(current_dt)
            nav_record = self.ledger.mark_to_market(current_dt, current_prices)
            current_nav = nav_record["total_nav"]

            if i % self.rebalance_frequency == 0 and i < len(oos_dates) - 1:

                hist_window = df[df["timestamp"] <= current_dt]
                piv_hist = hist_window.pivot(index="timestamp", columns="symbol", values="close").dropna()

                if len(piv_hist) >= 30:
                    ret_matrix = piv_hist[self.symbols].pct_change().dropna().values
                    recent_rets = ret_matrix[-63:]
                    cov_matrix = np.cov(recent_rets, rowvar=False)

                    today_feature = np.array([
                        float(np.mean(recent_rets[-1])),
                        float(np.std(recent_rets[-1]) * np.sqrt(252))
                    ])
                    raw_state = self.regime_detector.filter_step(today_feature, current_dt)
                    regime_state = self.hysteresis.filter(raw_state)

                    self.regimes_history.append({
                        "timestamp": current_dt,
                        "regime_id": regime_state.regime_id,
                        "regime_name": regime_state.regime_name,
                        "is_transition": regime_state.is_transition,
                        "probabilities": regime_state.probabilities
                    })

                    composite_views = {}
                    for sig in self.signals:
                        views = sig.generate_views(hist_window, current_dt, self.symbols)
                        for sym, v in views.items():
                            if sym not in composite_views:
                                composite_views[sym] = v
                            else:

                                prev = composite_views[sym]
                                blended_ret = 0.5 * prev.expected_return + 0.5 * v.expected_return
                                blended_conf = 0.5 * prev.confidence + 0.5 * v.confidence
                                composite_views[sym] = v

                    mu_post, V_post = self.bl.compute_posterior(
                        symbols=self.symbols,
                        covariance=cov_matrix,
                        views=composite_views,
                        regime=regime_state
                    )

                    target_w = self.optimizer.optimize(
                        mu=mu_post,
                        covariance=V_post,
                        current_weights=current_target_weights
                    )

                    scaled_w, cash_w, port_vol = self.vol_targeting.apply_vol_target(target_w, V_post)

                    turnover_step = float(np.sum(np.abs(scaled_w - current_target_weights)))
                    self.turnover_accumulator += turnover_step
                    current_target_weights = scaled_w

                    weight_entry = {"timestamp": current_dt, "cash": cash_w}
                    for s_idx, sym in enumerate(self.symbols):
                        weight_entry[sym] = float(scaled_w[s_idx])
                    self.weights_history.append(weight_entry)

                    for s_idx, sym in enumerate(self.symbols):
                        target_dollar = current_nav * scaled_w[s_idx]
                        price_now = current_prices.get(sym, 100.0)
                        target_shares = target_dollar / max(price_now, 1e-4)

                        current_shares = self.ledger.positions[sym].quantity if sym in self.ledger.positions else 0.0
                        delta_shares = target_shares - current_shares

                        if abs(delta_shares) >= 1.0:
                            side = OrderSide.BUY if delta_shares > 0 else OrderSide.SELL
                            order = Order(
                                order_id=f"ord_{i}_{sym}",
                                symbol=sym,
                                side=side,
                                quantity=abs(delta_shares),
                                order_type=OrderType.MARKET,
                                created_at=current_dt
                            )
                            pending_orders.append(order)

        history_df = self.ledger.get_history_df()
        years = len(oos_dates) / 252.0
        ann_turnover = self.turnover_accumulator / max(years, 0.1)

        from ramp.backtest.metrics import PerformanceMetricsCalculator
        metrics = PerformanceMetricsCalculator.calculate_all(
            history_df["daily_return"],
            turnover_annual=ann_turnover
        )

        return {
            "metrics": metrics,
            "history": history_df,
            "weights": pd.DataFrame(self.weights_history),
            "regimes": pd.DataFrame(self.regimes_history),
            "fills": self.ledger.fills_history,
        }
