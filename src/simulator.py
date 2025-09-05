# src/simulator.py
from __future__ import annotations
from typing import Tuple, List
import numpy as np

from .common import SimConfig, make_rng, draw_distances_and_prop


def simulate_ipact_offline_once(cfg: SimConfig, seed: int) -> Tuple[float, List[float]]:
    """
    Proportional-window model for Fig.4 (IPACT Limited, OFFLINE):
    - One fixed idle gap per cycle ≈ min(prop) + max(prop) - Guard_s.
    - Contiguous bursts thereafter (no per-ONU idle).
    - Total send time per cycle = rho_total * T_max  (window proportional to load).
    - Add total guard time: (N-1) * Guard_s.
    Idle Ratio = idle_fixed / (idle_fixed + (N-1)*Guard + rho*T_max).
    """
    rng = make_rng(seed)
    _dist_km, prop = draw_distances_and_prop(cfg.N, cfg.distance_km_min, cfg.distance_km_max, rng)

    prop_first = float(np.min(prop))
    prop_last  = float(np.max(prop))
    idle_fixed_s = max(0.0, prop_first + prop_last - cfg.Guard_s)
    guard_total_s = (cfg.N - 1) * cfg.Guard_s          # guards between N bursts
    send_time_s = cfg.rho_total * cfg.T_max_s    # proportional window to offered load

    cycle_len = idle_fixed_s + guard_total_s + send_time_s
    if cycle_len < 1e-12:
        cycle_len = 1e-12
    idle_ratio = idle_fixed_s / cycle_len

    idle_ratio_per_cycle: List[float] = []
    for k in range(cfg.cycles):
        if k >= cfg.warmup:
            idle_ratio_per_cycle.append(float(idle_ratio))

    mean_idle_ratio = float(np.mean(idle_ratio_per_cycle)) if idle_ratio_per_cycle else 0.0
    return mean_idle_ratio, idle_ratio_per_cycle


def run_replications(cfg: SimConfig, runs: int) -> Tuple[float, float]:
    """
    Run multiple independent replications (random distances change idle_fixed_s).
    Returns: (mean_of_rep_means, sample_std_of_rep_means)
    """
    if runs <= 0:
        return 0.0, 0.0
    rep_means = np.empty(runs, dtype=np.float64)
    for r in range(runs):
        m, _ = simulate_ipact_offline_once(cfg, seed=cfg.seed + r)
        rep_means[r] = m
    mean = float(rep_means.mean())
    std = float(rep_means.std(ddof=1)) if runs > 1 else 0.0
    return mean, std