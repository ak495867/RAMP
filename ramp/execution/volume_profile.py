from typing import Dict, List, Optional
import numpy as np


class IntradayVolumeProfiler:

    DEFAULT_30MIN_WEIGHTS = np.array([
        0.18, 0.12, 0.08, 0.06, 0.05, 0.04, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.17
    ])

    def __init__(self, custom_profile: Optional[np.ndarray] = None):
        if custom_profile is not None:
            norm_profile = np.array(custom_profile, dtype=float)
            self.profile = norm_profile / np.sum(norm_profile)
        else:
            self.profile = self.DEFAULT_30MIN_WEIGHTS / np.sum(self.DEFAULT_30MIN_WEIGHTS)

    def get_bin_count(self) -> int:
        return len(self.profile)

    def get_volume_fraction(self, bin_idx: int) -> float:
        if 0 <= bin_idx < len(self.profile):
            return float(self.profile[bin_idx])
        return 0.0

    def get_cumulative_profile(self) -> np.ndarray:
        return np.cumsum(self.profile)

    def fit_from_intraday_bars(self, intraday_volume_matrix: np.ndarray) -> "IntradayVolumeProfiler":
        mean_volumes = np.mean(intraday_volume_matrix, axis=0)
        total_vol = np.sum(mean_volumes)
        if total_vol > 0:
            self.profile = mean_volumes / total_vol
        return self


class UShapedVWAPSlicer:

    def __init__(self, profiler: Optional[IntradayVolumeProfiler] = None):
        self.profiler = profiler or IntradayVolumeProfiler()

    def generate_vwap_schedule(
        self,
        total_shares: float,
        adv_20: float,
        urgency_decay: float = 0.05
    ) -> List[Dict[str, float]]:
        if total_shares <= 0:
            return []

        profile = self.profiler.profile
        n_bins = len(profile)

        time_decay = np.exp(-urgency_decay * np.arange(n_bins))
        hybrid_weights = profile * time_decay
        hybrid_weights /= np.sum(hybrid_weights)

        scheduled_shares = hybrid_weights * total_shares
        remaining = total_shares

        time_labels = [
            "09:30-10:00", "10:00-10:30", "10:30-11:00", "11:00-11:30",
            "11:30-12:00", "12:00-12:30", "12:30-13:00", "13:00-13:30",
            "13:30-14:00", "14:00-14:30", "14:30-15:00", "15:00-15:30", "15:30-16:00"
        ]

        schedule = []
        for i in range(n_bins):
            qty = float(scheduled_shares[i])
            remaining = max(0.0, remaining - qty)
            expected_bin_market_volume = adv_20 * profile[i]
            participation_rate = qty / max(expected_bin_market_volume, 1.0)

            schedule.append({
                "interval_index": i,
                "time_window": time_labels[i] if i < len(time_labels) else f"interval_{i}",
                "volume_weight_pct": round(float(profile[i] * 100.0), 2),
                "scheduled_shares": round(qty, 2),
                "remaining_shares": round(float(remaining), 2),
                "participation_pct": round(float(participation_rate * 100.0), 3)
            })

        return schedule
