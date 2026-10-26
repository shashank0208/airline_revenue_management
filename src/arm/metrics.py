"""
Metrics for airline revenue management simulation.

Computes:
- Total revenue
- Load factor (seats sold / capacity)
- Spill (demand that was rejected)
- Paired comparison statistics (vs FCFS baseline)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PolicyMetrics:
    """Metrics for one policy across multiple Monte Carlo runs."""
    policy_name: str
    revenues: np.ndarray       # revenue per run
    seats_sold: np.ndarray     # seats sold per run
    spill: np.ndarray          # spilled requests per run
    capacity: int

    @property
    def mean_revenue(self) -> float:
        return float(np.mean(self.revenues))

    @property
    def std_revenue(self) -> float:
        return float(np.std(self.revenues, ddof=1))

    @property
    def ci_95(self) -> tuple[float, float]:
        """95% confidence interval for mean revenue."""
        n = len(self.revenues)
        se = self.std_revenue / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        return (self.mean_revenue - t_crit * se, self.mean_revenue + t_crit * se)

    @property
    def mean_load_factor(self) -> float:
        return float(np.mean(self.seats_sold / self.capacity))

    @property
    def mean_spill(self) -> float:
        return float(np.mean(self.spill))


@dataclass
class PairedComparison:
    """Paired comparison of a policy against a baseline (FCFS)."""
    policy_name: str
    baseline_name: str
    mean_uplift_pct: float          # mean (policy - baseline) / baseline * 100
    win_rate: float                  # fraction of runs where policy > baseline
    t_stat: float
    p_value_ttest: float
    wilcoxon_stat: float
    p_value_wilcoxon: float


def compute_paired_comparison(
    policy: PolicyMetrics,
    baseline: PolicyMetrics,
) -> PairedComparison:
    """Paired comparison between two policies on the same random streams."""
    diff = policy.revenues - baseline.revenues
    n = len(diff)

    # Mean percentage uplift
    base_mean = np.mean(baseline.revenues)
    uplift_pct = float(np.mean(diff) / base_mean * 100) if base_mean > 0 else 0.0

    # Win rate
    win_rate = float(np.mean(diff > 0))

    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(policy.revenues, baseline.revenues)

    # Wilcoxon signed-rank test (requires non-zero differences)
    nonzero = diff[diff != 0]
    if len(nonzero) >= 10:
        w_stat, p_wilcoxon = stats.wilcoxon(nonzero)
    else:
        w_stat, p_wilcoxon = 0.0, 1.0

    return PairedComparison(
        policy_name=policy.policy_name,
        baseline_name=baseline.policy_name,
        mean_uplift_pct=uplift_pct,
        win_rate=win_rate,
        t_stat=float(t_stat),
        p_value_ttest=float(p_ttest),
        wilcoxon_stat=float(w_stat),
        p_value_wilcoxon=float(p_wilcoxon),
    )
