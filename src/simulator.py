# src/simulator.py
from __future__ import annotations
from typing import Tuple, List
import numpy as np

from src.common import (
    SimConfig,
    make_rng,
    draw_distances_and_prop,
    per_onu_cap_bytes,
)

def simulate_ipact_offline_once(cfg: SimConfig, seed: int) -> Tuple[float, List[float]]:
    """
    One replication of EPON upstream under IPACT (Limited) with OFFLINE scheduling, fluid arrivals.

    Returns:
        (mean_idle_ratio_over_steady_cycles, idle_ratio_per_cycle_list)
    """
    rng = make_rng(seed)
    _dist_km, prop = draw_distances_and_prop(
        cfg.N, cfg.distance_km_min, cfg.distance_km_max, rng
    )
    Wmax_bytes = per_onu_cap_bytes(cfg.R_bps, cfg.T_max_s, cfg.N)
    lambda_Bps_each = (cfg.rho_total * cfg.R_bps / cfg.N) / 8.0

    N = cfg.N
    backlog = np.zeros(N, dtype=np.float64)
    grants  = np.zeros(N, dtype=np.float64)

    T_collect_prev = 0.0
    cycle_start = 0.0
    idle_ratio_per_cycle: List[float] = []

    for k in range(cfg.cycles):
        end_prev = cycle_start
        cycle_idle = 0.0

        ends = np.empty(N, dtype=np.float64)
        report_arrivals = np.empty(N, dtype=np.float64)

        # ---- schedule actual transmissions for cycle k (OFFLINE gates) ----
        for i in range(N):
            baseline = end_prev + cfg.Guard_s
            gate_arrival = T_collect_prev + prop[i]
            start_i = baseline if baseline >= gate_arrival else gate_arrival
            idle_i = start_i - baseline if start_i > baseline else 0.0

            tx_time = (grants[i] * 8.0) / cfg.R_bps
            end_i = start_i + tx_time

            cycle_idle += idle_i
            end_prev = end_i

            ends[i] = end_i
            report_arrivals[i] = end_i + prop[i]

            # subtract transmitted bytes; clamp to 0
            sent = grants[i]
            b = backlog[i] - sent
            backlog[i] = b if b > 0.0 else 0.0

        cycle_end = end_prev
        cycle_len = cycle_end - cycle_start
        if cycle_len < 1e-12:
            cycle_len = 1e-12

        idle_ratio = cycle_idle / cycle_len
        if k >= cfg.warmup:
            idle_ratio_per_cycle.append(idle_ratio)

        # ---- collect reports for next cycle ----
        T_collect = float(report_arrivals.max())

        # arrivals continue until T_collect (precise per-ONU using end_i)
        add_time = T_collect - ends
        add_time[add_time < 0.0] = 0.0
        backlog += lambda_Bps_each * add_time

        # ---- compute next-cycle grants (Limited) ----
        # vectorized min with scalar Wmax
        np.minimum(backlog, Wmax_bytes, out=grants)

        # ---- advance ----
        cycle_start = cycle_end
        T_collect_prev = T_collect

    mean_idle_ratio = float(np.mean(idle_ratio_per_cycle)) if idle_ratio_per_cycle else 0.0
    # convert list elements to plain float for cleaner serialization/printing
    return mean_idle_ratio, list(map(float, idle_ratio_per_cycle))

def run_replications(cfg: SimConfig, runs: int) -> Tuple[float, float]:
    """
    Run multiple independent replications.
    Returns:
        (mean_of_rep_means, sample_std_of_rep_means)
    """
    if runs <= 0:
        return 0.0, 0.0
    rep_means = np.empty(runs, dtype=np.float64)
    for r in range(runs):
        seed_r = cfg.seed + r
        m, _ = simulate_ipact_offline_once(cfg, seed_r)
        rep_means[r] = m
    mean = float(rep_means.mean())
    std = float(rep_means.std(ddof=1)) if runs > 1 else 0.0
    return mean, std
