"""Threshold derivation per PLAN Lemma 2.4.

    eta = epsilon / R_max - exp(delta_avg)
    tau(K, eta) = eta + sqrt( log(2/alpha) / (2K) )    # DKW band

`tau` is reused for both the Share check and the Prune check.  Setting
`use_dkw=False` falls back to plain eta (useful for ablations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThresholdConfig:
    epsilon: float = 0.02      # acceptable value error
    r_max: float = 1.0         # max reward magnitude
    gamma: float = 0.5         # discount factor for TV -> value error bound
    alpha: float = 0.05        # DKW confidence level
    K: int = 10                # fast subset size
    use_dkw: bool = True
    eta_override: Optional[float] = None  # for ablation: bypass Lemma 2.4


def compute_eta(cfg: ThresholdConfig, delta_avg: float = 0.0) -> float:
    """eta from Lemma 2.4. delta_avg = exp-mean of per-segment delta."""

    if cfg.eta_override is not None:
        return float(cfg.eta_override)
    eta = cfg.epsilon / max(cfg.r_max, 1e-8) - delta_avg
    return max(eta, 1e-6)


def compute_tau(cfg: ThresholdConfig, eta: float) -> float:
    if not cfg.use_dkw:
        return eta
    band = math.sqrt(math.log(2.0 / cfg.alpha) / (2.0 * max(cfg.K, 1)))
    return eta + band


def tv_to_value_bound(tv: float, cfg: ThresholdConfig) -> float:
    """Convert total variation to the |V_x - V_y| bound.

    The theoretical formula is:

        1 / (1 - gamma) - 1 / (1 - gamma * (1 - epsilon_T / 2))

    where epsilon_T / 2 is the total variation distance TV(x, y).  The
    ``tv`` argument is therefore substituted directly for epsilon_T / 2.
    """

    gamma = min(max(float(cfg.gamma), 0.0), 1.0 - 1e-8)
    tv = max(float(tv), 0.0)
    return (1.0 / (1.0 - gamma)) - (1.0 / (1.0 - gamma * (1.0 - tv)))
