"""
Anti-Whipsaw Hysteresis and Minimum Dwell Time Filter.
Prevents false regime flips, suppresses turnover explosion, and protects alpha.
"""

from datetime import datetime
from typing import Dict, Optional
import numpy as np
from ramp.core.types import RegimeState


class RegimeHysteresisFilter:
    """
    Applies state-persistence constraints and hysteresis thresholds to raw regime probabilities.
    
    Prevents portfolio turnover churn when probabilities hover around 50/50.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        min_dwell_bars: int = 3,
        crisis_fast_trigger: bool = True,
        crisis_regime_id: int = 2
    ):
        self.confidence_threshold = confidence_threshold
        self.min_dwell_bars = min_dwell_bars
        self.crisis_fast_trigger = crisis_fast_trigger
        self.crisis_regime_id = crisis_regime_id

        self.current_regime: Optional[int] = None
        self.dwell_count: int = 0
        self.last_timestamp: Optional[datetime] = None

    def filter(
        self,
        raw_state: RegimeState,
        regime_names: Optional[Dict[int, str]] = None
    ) -> RegimeState:
        """
        Filters raw state probabilities and returns the stabilized RegimeState.
        """
        probs = raw_state.probabilities
        best_candidate = max(probs, key=probs.get)
        best_prob = probs[best_candidate]

        # Initialization
        if self.current_regime is None:
            self.current_regime = best_candidate
            self.dwell_count = 1
            self.last_timestamp = raw_state.timestamp
            return RegimeState(
                timestamp=raw_state.timestamp,
                regime_id=self.current_regime,
                regime_name=raw_state.regime_name,
                probabilities=probs,
                is_transition=True,
                entropy=raw_state.entropy
            )

        self.dwell_count += 1
        is_transition = False

        if best_candidate != self.current_regime:
            # Check crisis fast-trigger bypass
            is_crisis_emergency = (
                self.crisis_fast_trigger and 
                best_candidate == self.crisis_regime_id and 
                best_prob >= 0.55
            )

            # Normal hysteresis condition: high confidence + minimum dwell
            can_switch = (
                best_prob >= self.confidence_threshold and 
                self.dwell_count > self.min_dwell_bars
            ) or is_crisis_emergency

            if can_switch:
                self.current_regime = best_candidate
                self.dwell_count = 1
                is_transition = True

        name = regime_names.get(self.current_regime, f"regime_{self.current_regime}") if regime_names else raw_state.regime_name

        return RegimeState(
            timestamp=raw_state.timestamp,
            regime_id=self.current_regime,
            regime_name=name,
            probabilities=probs,
            is_transition=is_transition,
            entropy=raw_state.entropy
        )

    def reset(self) -> None:
        """Resets the filter state."""
        self.current_regime = None
        self.dwell_count = 0
        self.last_timestamp = None
