from dataclasses import dataclass
import os
import numpy as np
from numpy.random import Generator
from typing import Tuple


@dataclass
class SimConfig:
    """Simulation configuration for EPON IPACT-Offline."""
    N: int = 32
    R_bps: float = 10e9
    Guard_s: float = 624e-9
    T_max_s: float = 1e-3
    distance_km_min: float = 10.0
    distance_km_max: float = 20.0
    rho_total: float = 0.8
    cycles: int = 3500
    warmup: int = 500
    seed: int = 123

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.N < 1:
            raise ValueError("N must be >= 1")
        if self.R_bps <= 0:
            raise ValueError("R_bps must be > 0")
        if self.Guard_s < 0:
            raise ValueError("Guard_s must be >= 0")
        if self.T_max_s <= 0:
            raise ValueError("T_max_s must be > 0")
        if not (0 <= self.rho_total <= 1):
            raise ValueError("rho_total must be in [0, 1]")
        if self.distance_km_min < 0 or self.distance_km_max < self.distance_km_min:
            raise ValueError("Invalid distance range: 0 <= distance_km_min <= distance_km_max")
        if self.cycles <= self.warmup or self.warmup < 0:
            raise ValueError("cycles must be > warmup >= 0")


def make_rng(seed: int) -> Generator:
    """Create a NumPy random number generator with the given seed.

    Args:
        seed: Integer seed for reproducibility.

    Returns:
        NumPy Generator object.
    """
    return np.random.default_rng(seed)


def draw_distances_and_prop(N: int, km_min: float, km_max: float, rng: Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ONU distances and compute one-way propagation delays.

    Args:
        N: Number of ONUs.
        km_min: Minimum distance in kilometers.
        km_max: Maximum distance in kilometers.
        rng: NumPy random number generator.

    Returns:
        Tuple of (distances_km, prop_s):
            - distances_km: Array of shape (N,) with distances in km (float64).
            - prop_s: Array of shape (N,) with one-way propagation delays in seconds (float64).
    """
    distances_km = rng.uniform(km_min, km_max, N).astype(np.float64)
    prop_s = distances_km * 5e-6
    return distances_km, prop_s


def per_onu_cap_bytes(R_bps: float, T_max_s: float, N: int) -> int:
    """Compute per-ONU byte cap for IPACT Limited policy.

    Args:
        R_bps: Line rate in bits per second.
        T_max_s: Maximum cycle time in seconds.
        N: Number of ONUs.

    Returns:
        Integer byte cap per ONU per cycle.

    Raises:
        ValueError: If R_bps <= 0, T_max_s <= 0, or N < 1.
    """
    if R_bps <= 0 or T_max_s <= 0 or N < 1:
        raise ValueError("R_bps and T_max_s must be > 0, N >= 1")
    return int(np.floor((R_bps * T_max_s) / N / 8))


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)