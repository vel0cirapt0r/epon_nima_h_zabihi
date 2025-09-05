# src/idle_ratio_ipact_offline_fig4_bars.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (imported for 3D projection)
from matplotlib import cm
from matplotlib.colors import Normalize

from src.common import SimConfig, ensure_dir
from src.simulator import run_replications


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_grid_df(
    rho_list: list[float],
    tmax_ms_list: list[float],
    runs: int,
    cycles: int,
    warmup: int,
    seed: int,
    N: int,
    Rb: float,
    guard_ns: float,
    dmin_km: float,
    dmax_km: float,
) -> pd.DataFrame:
    rows = []
    for rho in rho_list:
        for tmax_ms in tmax_ms_list:
            cfg = SimConfig(
                N=N,
                R_bps=Rb,
                Guard_s=guard_ns * 1e-9,
                T_max_s=tmax_ms / 1000.0,
                distance_km_min=dmin_km,
                distance_km_max=dmax_km,
                rho_total=rho,
                cycles=cycles,
                warmup=warmup,
                seed=seed,
            )
            mean, std = run_replications(cfg, runs)
            pct = 100.0 * mean
            rows.append(
                {
                    "rho_total": rho,
                    "T_max_ms": tmax_ms,
                    "idle_ratio_mean": float(mean),
                    "idle_ratio_std": float(std),
                    "idle_ratio_pct": float(pct),
                }
            )
            print(f"rho={rho:.2f}, T_max={tmax_ms:.1f} ms -> idle={mean:.4f} ({pct:.2f}%)", flush=True)
    return pd.DataFrame(rows)


def plot_panels4(df: pd.DataFrame, out_path: str) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    tmax_sorted = [1, 2, 5, 10]  # match the paper’s panels
    for i, tmax in enumerate(tmax_sorted):
        subdf = df[df["T_max_ms"] == tmax].sort_values("rho_total")
        ax = axs[i]
        ax.bar(subdf["rho_total"], subdf["idle_ratio_pct"], width=0.08)
        ax.set_title(f"T_max = {tmax} ms")
        ax.set_xlabel("ONUs load (ρ)")
        ax.set_xticks(list(subdf["rho_total"]))
        ax.set_ylim(0, 45)
        ax.grid(True, linestyle="--", alpha=0.5)
    axs[0].set_ylabel("Average idle ratio (%)")
    fig.suptitle("Figure 4 – IPACT (Offline): Average idle ratio (bars)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_3d(df: pd.DataFrame, out_path: str) -> None:
    # Axes: X -> T_max_ms (1,2,5,10), Y -> rho_total (0.2..1.0), Z -> idle_ratio_pct
    tmax_unique = sorted({float(x) for x in df["T_max_ms"].unique()})
    rho_unique = sorted({float(x) for x in df["rho_total"].unique()})
    nt, nr = len(tmax_unique), len(rho_unique)

    # Build Z values in row-major order: for each rho, iterate over all tmax
    top_vals: list[float] = []
    for r in rho_unique:
        for t in tmax_unique:
            v = df[(df["rho_total"] == r) & (df["T_max_ms"] == t)]["idle_ratio_pct"].iloc[0]
            top_vals.append(float(v))
    top = np.array(top_vals, dtype=float)

    _xx, _yy = np.meshgrid(range(nt), range(nr))  # shapes (nr, nt)
    x, y = _xx.ravel(), _yy.ravel()
    z0 = np.zeros_like(top)
    dx = dy = 0.8

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    norm = Normalize(vmin=float(top.min()), vmax=float(top.max()))
    colors = cm.viridis(norm(top))
    ax.bar3d(x, y, z0, dx, dy, top, shade=True, color=colors)

    ax.set_xticks(range(nt))
    ax.set_xticklabels([str(int(t)) for t in tmax_unique])
    ax.set_xlabel("Maximum cycle length [ms]")

    ax.set_yticks(range(nr))
    ax.set_yticklabels([f"{r:.1f}" for r in rho_unique])
    ax.set_ylabel("ONUs load")

    ax.set_zlabel("Average idle ratio [%]")
    ax.set_zlim(0, 45)

    sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label("Average idle ratio [%]")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate bar plots for Figure 4 – IPACT (Offline)")
    parser.add_argument("--rho_list", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--tmax_list", type=str, default="1,2,5,10")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--cycles", type=int, default=3500)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=str, default="out/fig4_idle_ratio_ipact_offline")
    parser.add_argument("--style", default="panels4", choices=["panels4", "3d", "both"])
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--Rb", type=float, default=10e9)
    parser.add_argument("--guard_ns", type=float, default=624)
    parser.add_argument("--dmin_km", type=float, default=10.0)
    parser.add_argument("--dmax_km", type=float, default=20.0)

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    rho_list = parse_float_list(args.rho_list)
    tmax_ms_list = parse_float_list(args.tmax_list)

    df = build_grid_df(
        rho_list=rho_list,
        tmax_ms_list=tmax_ms_list,
        runs=args.runs,
        cycles=args.cycles,
        warmup=args.warmup,
        seed=args.seed,
        N=args.N,
        Rb=args.Rb,
        guard_ns=args.guard_ns,
        dmin_km=args.dmin_km,
        dmax_km=args.dmax_km,
    )
    df.to_csv(os.path.join(args.out_dir, "fig4_grid.csv"), index=False)

    if args.style in ("panels4", "both"):
        plot_panels4(df, os.path.join(args.out_dir, "fig4_bars_4panels.png"))

    if args.style in ("3d", "both"):
        plot_3d(df, os.path.join(args.out_dir, "fig4_bars_3d.png"))


if __name__ == "__main__":
    main()
