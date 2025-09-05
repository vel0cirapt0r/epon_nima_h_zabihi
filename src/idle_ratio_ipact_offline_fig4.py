# src/idle_ratio_ipact_offline_fig4.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

from src.common import SimConfig, ensure_dir
from src.simulator import run_replications


def parse_float_list(s: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def run_panel_A(rho: float, tmax_ms_list: list[float],
                runs: int, cycles: int, warmup: int, seed: int, out_dir: str) -> None:
    rows = []
    for tmax_ms in tmax_ms_list:
        cfg = SimConfig(
            N=32,
            R_bps=10e9,
            Guard_s=624e-9,
            T_max_s=tmax_ms / 1000.0,
            distance_km_min=10.0,
            distance_km_max=20.0,
            rho_total=rho,
            cycles=cycles,
            warmup=warmup,
            seed=seed,
        )
        mean_idle, std_idle = run_replications(cfg, runs)
        rows.append(
            {
                "T_max_ms": tmax_ms,
                "idle_ratio_mean": mean_idle,
                "idle_ratio_std": std_idle,
            }
        )
        print(f"[A] T_max={tmax_ms:.1f} ms → idle={mean_idle:.6f} (std={std_idle:.6f})")

    df = pd.DataFrame(rows)
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "fig4_panelA.csv")
    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(df["T_max_ms"], df["idle_ratio_mean"], marker="o")
    plt.xlabel("T_max (ms)")
    plt.ylabel("Average Idle Ratio")
    plt.title("Figure 4 – IPACT (Offline): Idle vs T_max")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_panelA.png"), dpi=160)
    plt.close()


def run_panel_B(tmax_ms: float, rho_list: list[float],
                runs: int, cycles: int, warmup: int, seed: int, out_dir: str) -> None:
    rows = []
    tmax_s = tmax_ms / 1000.0
    for rho in rho_list:
        cfg = SimConfig(
            N=32,
            R_bps=10e9,
            Guard_s=624e-9,
            T_max_s=tmax_s,
            distance_km_min=10.0,
            distance_km_max=20.0,
            rho_total=rho,
            cycles=cycles,
            warmup=warmup,
            seed=seed,
        )
        mean_idle, std_idle = run_replications(cfg, runs)
        rows.append(
            {
                "rho_total": rho,
                "idle_ratio_mean": mean_idle,
                "idle_ratio_std": std_idle,
            }
        )
        print(f"[B] rho={rho:.2f} → idle={mean_idle:.6f} (std={std_idle:.6f})")

    df = pd.DataFrame(rows)
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "fig4_panelB.csv")
    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(df["rho_total"], df["idle_ratio_mean"], marker="o")
    plt.xlabel("Aggregated Load (ρ)")
    plt.ylabel("Average Idle Ratio")
    plt.title("Figure 4 – IPACT (Offline): Idle vs Load")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_panelB.png"), dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 4 – IPACT (Offline) Average Idle Ratio"
    )
    parser.add_argument("panel", choices=["A", "B"], help="Panel to generate: A or B")
    parser.add_argument("--runs", type=int, default=50, help="Replications per point")
    parser.add_argument("--cycles", type=int, default=3500, help="Cycles per replication")
    parser.add_argument("--warmup", type=int, default=500, help="Warm-up cycles to discard")
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed")

    # Panel A
    parser.add_argument("--rho", type=float, default=0.8, help="Aggregated load for Panel A")
    parser.add_argument(
        "--tmax_list",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated T_max values (ms) for Panel A",
    )

    # Panel B
    parser.add_argument("--tmax", type=float, default=1.0, help="T_max (ms) for Panel B")
    parser.add_argument(
        "--rho_list",
        type=str,
        default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated ρ values for Panel B",
    )

    args = parser.parse_args()

    out_dir = os.path.join("out", "fig4_idle_ratio_ipact_offline")
    ensure_dir(out_dir)

    if args.panel == "A":
        tmax_ms_list = parse_float_list(args.tmax_list)
        if not tmax_ms_list:
            raise ValueError("Empty tmax_list after parsing.")
        run_panel_A(args.rho, tmax_ms_list, args.runs, args.cycles, args.warmup, args.seed, out_dir)
    else:
        rho_list = parse_float_list(args.rho_list)
        if not rho_list:
            raise ValueError("Empty rho_list after parsing.")
        run_panel_B(args.tmax, rho_list, args.runs, args.cycles, args.warmup, args.seed, out_dir)


if __name__ == "__main__":
    main()
