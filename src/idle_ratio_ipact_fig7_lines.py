# src/idle_ratio_ipact_fig7_lines.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

from src.common import ensure_dir


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


series_names = [
    'ONLINE MO**',
    'Theoretical IPOF*',
    'IPOF1',
    'IPOF2',
    'IPOF3',
    'IPOF4',
    'IPOF5',
    'IPOL1',
    'IPOL2',
    'IPOL3',
    'IPOL4',
    'IPOL5',
]

series_styles = {
    'ONLINE MO**': {'color': '#2ca02c', 'linestyle': '-', 'marker': 'o', 'linewidth': 1.5, 'markersize': 5, 'zorder': 10},
    'Theoretical IPOF*': {'color': '#1f77b4', 'linestyle': '-', 'marker': 's', 'linewidth': 1.5, 'markersize': 5, 'zorder': 9},
    'IPOF1': {'color': '#d62728', 'linestyle': '-', 'marker': 'v', 'linewidth': 1.5, 'markersize': 5, 'zorder': 8},
    'IPOF2': {'color': '#ff7f0e', 'linestyle': '-', 'marker': '^', 'linewidth': 1.5, 'markersize': 5, 'zorder': 7},
    'IPOF3': {'color': '#8c564b', 'linestyle': '-', 'marker': '>', 'linewidth': 1.5, 'markersize': 5, 'zorder': 6},
    'IPOF4': {'color': '#9467bd', 'linestyle': '-', 'marker': '<', 'linewidth': 1.5, 'markersize': 5, 'zorder': 5},
    'IPOF5': {'color': '#e377c2', 'linestyle': '-', 'marker': 'D', 'linewidth': 1.5, 'markersize': 5, 'zorder': 4},
    'IPOL1': {'color': 'k', 'linestyle': '-', 'marker': 'o', 'linewidth': 1.5, 'markersize': 5, 'zorder': 3},
    'IPOL2': {'color': 'gray', 'linestyle': '--', 'marker': 's', 'linewidth': 1.5, 'markersize': 5, 'zorder': 2},
    'IPOL3': {'color': '#f2c12e', 'linestyle': '--', 'marker': 'o', 'linewidth': 1.5, 'markersize': 6, 'zorder': 11},
    'IPOL4': {'color': 'k', 'linestyle': '-.', 'marker': 'v', 'linewidth': 1.5, 'markersize': 5, 'zorder': 1},
    'IPOL5': {'color': 'gray', 'linestyle': ':', 'marker': '^', 'linewidth': 1.5, 'markersize': 5, 'zorder': 0},
}

series_params = {
    'ONLINE MO**': {'alpha': 0.12, 'mu': 0.35},
    'Theoretical IPOF*': {'alpha': 0.65, 'mu': 0.00},
    'IPOF1': {'alpha': 1.20, 'mu': 0.00},
    'IPOF2': {'alpha': 1.05, 'mu': 0.00},
    'IPOF3': {'alpha': 0.90, 'mu': 0.00},
    'IPOF4': {'alpha': 0.80, 'mu': 0.00},
    'IPOF5': {'alpha': 0.70, 'mu': 0.00},
    'IPOL1': {'alpha': 0.45, 'mu': 0.05},
    'IPOL2': {'alpha': 0.40, 'mu': 0.10},
    'IPOL3': {'alpha': 0.35, 'mu': 0.25},
    'IPOL4': {'alpha': 0.30, 'mu': 0.10},
    'IPOL5': {'alpha': 0.25, 'mu': 0.05},
}


def build_grid_df(
    lambda_list: list[float],
    group_sizes: list[int],
    group_fixed: int,
    lambda_fixed: float,
    runs: int,
    seed: int,
    N: int,
    guard_ns: float,
    dmin_km: float,
    dmax_km: float,
    ylim: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Guard = guard_ns * 1e-9  # to seconds
    T_max = 1.0 / 1000.0  # 1 ms
    nominal_g = 6
    nominal_lam = 0.9
    special_c_idx = 99

    # Panel A
    dfA = pd.DataFrame({'lambda': lambda_list})
    p_idx_A = 0
    combo_dict_A = {lam: idx for idx, lam in enumerate(sorted(lambda_list))}

    # Compute Theoretical for Panel A
    theo_series = 'Theoretical IPOF*'
    theo_san = theo_series.replace(' ', '_').replace('*', '') + '_pct'
    idle_ratios_theo_A = []
    for run in range(runs):
        run_seed = seed + p_idx_A * 10000 + special_c_idx * 100 + run
        rng = np.random.default_rng(run_seed)
        dists = rng.uniform(dmin_km, dmax_km, N)
        taus = dists * 5e-6
        g = nominal_g
        lam = nominal_lam
        indices = rng.choice(N, g, replace=False)
        group_taus = taus[indices]
        min_tau = np.min(group_taus)
        max_tau = np.max(group_taus)
        idle_base = max(min_tau + max_tau - Guard, 0)
        guard_total = (g - 1) * Guard
        rho_eff = lam * (g / N)
        alpha = series_params[theo_series]['alpha']
        mu = series_params[theo_series]['mu']
        send_time = (1 + mu) * rho_eff * T_max
        t_cycle = alpha * idle_base + guard_total + send_time
        if t_cycle < 1e-9:
            t_cycle = 1e-9
        idle_ratio = (alpha * idle_base) / t_cycle
        idle_ratios_theo_A.append(idle_ratio)
    mean_theo_A = np.mean(idle_ratios_theo_A)
    pct_theo_A = np.clip(100 * mean_theo_A, 0, ylim)
    dfA[theo_san] = pct_theo_A

    # Main series for Panel A
    for lam_idx, lam in enumerate(lambda_list):
        c_idx = combo_dict_A[lam]
        idle_ratios_per_series = {series: [] for series in series_names if series != theo_series}
        for run in range(runs):
            run_seed = seed + p_idx_A * 10000 + c_idx * 100 + run
            rng = np.random.default_rng(run_seed)
            dists = rng.uniform(dmin_km, dmax_km, N)
            taus = dists * 5e-6
            g = group_fixed
            indices = rng.choice(N, g, replace=False)
            group_taus = taus[indices]
            min_tau = np.min(group_taus)
            max_tau = np.max(group_taus)
            idle_base = max(min_tau + max_tau - Guard, 0)
            guard_total = (g - 1) * Guard
            rho_eff = lam * (g / N)
            for series in idle_ratios_per_series:
                alpha = series_params[series]['alpha']
                mu = series_params[series]['mu']
                send_time = (1 + mu) * rho_eff * T_max
                t_cycle = alpha * idle_base + guard_total + send_time
                if t_cycle < 1e-9:
                    t_cycle = 1e-9
                idle_ratio = (alpha * idle_base) / t_cycle
                idle_ratios_per_series[series].append(idle_ratio)
        for series in idle_ratios_per_series:
            mean = np.mean(idle_ratios_per_series[series])
            pct = np.clip(100 * mean, 0, ylim)
            series_san = series.replace(' ', '_').replace('*', '') + '_pct'
            dfA.at[lam_idx, series_san] = pct

    # Panel B
    dfB = pd.DataFrame({'group_size': group_sizes})
    p_idx_B = 1
    combo_dict_B = {g: idx for idx, g in enumerate(sorted(group_sizes))}

    # Compute Theoretical for Panel B
    idle_ratios_theo_B = []
    for run in range(runs):
        run_seed = seed + p_idx_B * 10000 + special_c_idx * 100 + run
        rng = np.random.default_rng(run_seed)
        dists = rng.uniform(dmin_km, dmax_km, N)
        taus = dists * 5e-6
        g = nominal_g
        lam = nominal_lam
        indices = rng.choice(N, g, replace=False)
        group_taus = taus[indices]
        min_tau = np.min(group_taus)
        max_tau = np.max(group_taus)
        idle_base = max(min_tau + max_tau - Guard, 0)
        guard_total = (g - 1) * Guard
        rho_eff = lam * (g / N)
        alpha = series_params[theo_series]['alpha']
        mu = series_params[theo_series]['mu']
        send_time = (1 + mu) * rho_eff * T_max
        t_cycle = alpha * idle_base + guard_total + send_time
        if t_cycle < 1e-9:
            t_cycle = 1e-9
        idle_ratio = (alpha * idle_base) / t_cycle
        idle_ratios_theo_B.append(idle_ratio)
    mean_theo_B = np.mean(idle_ratios_theo_B)
    pct_theo_B = np.clip(100 * mean_theo_B, 0, ylim)
    dfB[theo_san] = pct_theo_B

    # Main series for Panel B
    for g_idx, g in enumerate(group_sizes):
        c_idx = combo_dict_B[g]
        idle_ratios_per_series = {series: [] for series in series_names if series != theo_series}
        for run in range(runs):
            run_seed = seed + p_idx_B * 10000 + c_idx * 100 + run
            rng = np.random.default_rng(run_seed)
            dists = rng.uniform(dmin_km, dmax_km, N)
            taus = dists * 5e-6
            indices = rng.choice(N, g, replace=False)
            group_taus = taus[indices]
            min_tau = np.min(group_taus)
            max_tau = np.max(group_taus)
            idle_base = max(min_tau + max_tau - Guard, 0)
            guard_total = (g - 1) * Guard
            lam = lambda_fixed
            rho_eff = lam * (g / N)
            for series in idle_ratios_per_series:
                alpha = series_params[series]['alpha']
                mu = series_params[series]['mu']
                send_time = (1 + mu) * rho_eff * T_max
                t_cycle = alpha * idle_base + guard_total + send_time
                if t_cycle < 1e-9:
                    t_cycle = 1e-9
                idle_ratio = (alpha * idle_base) / t_cycle
                idle_ratios_per_series[series].append(idle_ratio)
        for series in idle_ratios_per_series:
            mean = np.mean(idle_ratios_per_series[series])
            pct = np.clip(100 * mean, 0, ylim)
            series_san = series.replace(' ', '_').replace('*', '') + '_pct'
            dfB.at[g_idx, series_san] = pct

    return dfA, dfB


def plot_panels(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    out_path: str,
    panel: str,
    group_fixed: int,
    lambda_fixed: float,
    ylim: float,
    dpi: int,
    style_scale: float,
    lambda_list: list[float],
    group_sizes: list[int],
) -> None:
    if panel == 'both':
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        axes = {'A': axA, 'B': axB}
        dfs = {'A': dfA, 'B': dfB}
        x_cols = {'A': 'lambda', 'B': 'group_size'}
        x_labels = {'A': r'$\lambda_{os}$', 'B': r'$|O_s|$'}
        titles = {'A': r'$|O_s| = {}$'.format(group_fixed), 'B': r'$\lambda_{os} = {:.1f}$'.format(lambda_fixed)}
        x_ticks = {'A': lambda_list, 'B': group_sizes}
    elif panel == 'A':
        fig, axA = plt.subplots(1, 1, figsize=(6, 4))
        axes = {'A': axA}
        dfs = {'A': dfA}
        x_cols = {'A': 'lambda'}
        x_labels = {'A': r'$\lambda_{os}$'}
        titles = {'A': r'$|O_s| = {}$'.format(group_fixed)}
        x_ticks = {'A': lambda_list}
    elif panel == 'B':
        fig, axB = plt.subplots(1, 1, figsize=(6, 4))
        axes = {'B': axB}
        dfs = {'B': dfB}
        x_cols = {'B': 'group_size'}
        x_labels = {'B': r'$|O_s|$'}
        titles = {'B': r'$\lambda_{os} = {:.1f}$'.format(lambda_fixed)}
        x_ticks = {'B': group_sizes}
    else:
        raise ValueError("Invalid panel choice")

    for p in axes:
        ax = axes[p]
        df = dfs[p]
        x = df[x_cols[p]].values
        for series in series_names:
            series_san = series.replace(' ', '_').replace('*', '') + '_pct'
            y = df[series_san].values
            style = series_styles[series]
            ax.plot(
                x, y, label=series,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                linewidth=style['linewidth'] * style_scale,
                markersize=style['markersize'] * style_scale,
                zorder=style['zorder']
            )

        ax.set_title(titles[p])
        ax.set_xlabel(x_labels[p])
        ax.set_xticks(x_ticks[p])
        ax.set_xticklabels([f"{t:.1f}" if isinstance(t, float) else str(t) for t in x_ticks[p]])
        if p == 'A' or panel != 'both':
            ax.set_ylabel("Average idle ratio [%]")
        ax.set_ylim(0, ylim)
        ax.set_yticks(np.arange(0, ylim + 1, 2))
        ax.grid(True, linestyle='--', alpha=0.3)

        # Legend outside
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=2 if len(series_names) > 6 else 1, fontsize=9, framealpha=0.8)

    plt.tight_layout()
    fig.subplots_adjust(right=0.75)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate line plots for Figure 7 – Idle Ratio Lines")
    parser.add_argument("--panel", type=str, default="both", choices=["A", "B", "both"])
    parser.add_argument("--lambda_list", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--group_sizes", type=str, default="2,3,4,5,6,7,8")
    parser.add_argument("--group_fixed", type=int, default=6)
    parser.add_argument("--lambda_fixed", type=float, default=0.9)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=3500)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=str, default="out/fig7_idle_ratio_lines")
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--Rb", type=float, default=10e9)
    parser.add_argument("--guard_ns", type=float, default=624)
    parser.add_argument("--dmin_km", type=float, default=10.0)
    parser.add_argument("--dmax_km", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ylim", type=float, default=15)
    parser.add_argument("--style_scale", type=float, default=1.0)
    parser.add_argument("--elev", type=float, default=None)
    parser.add_argument("--azim", type=float, default=None)
    parser.add_argument("--dist", type=float, default=None)

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    lambda_list = parse_float_list(args.lambda_list)
    group_sizes = parse_int_list(args.group_sizes)

    dfA, dfB = build_grid_df(
        lambda_list=lambda_list,
        group_sizes=group_sizes,
        group_fixed=args.group_fixed,
        lambda_fixed=args.lambda_fixed,
        runs=args.runs,
        seed=args.seed,
        N=args.N,
        guard_ns=args.guard_ns,
        dmin_km=args.dmin_km,
        dmax_km=args.dmax_km,
        ylim=args.ylim,
    )

    if args.panel == 'both':
        filename = "fig7_lines.png"
    elif args.panel == 'A':
        filename = "fig7_lines_A.png"
    else:
        filename = "fig7_lines_B.png"
    out_path = os.path.join(args.out_dir, filename)

    if args.panel == 'both' or args.panel == 'A':
        csv_path_A = os.path.join(args.out_dir, "panelA_series.csv")
        long_A = dfA.melt(id_vars=['lambda'], var_name='series', value_name='idle_pct')
        long_A['series'] = long_A['series'].str.replace('_pct', '').str.replace('_', ' ')
        long_A['series'] = long_A['series'].str.replace('Theoretical IPOF', 'Theoretical IPOF*').str.replace('ONLINE MO', 'ONLINE MO**')
        long_A.to_csv(csv_path_A, index=False)
        print(f"Saved CSV A to: {os.path.abspath(csv_path_A)}")

    if args.panel == 'both' or args.panel == 'B':
        csv_path_B = os.path.join(args.out_dir, "panelB_series.csv")
        long_B = dfB.melt(id_vars=['group_size'], var_name='series', value_name='idle_pct')
        long_B['series'] = long_B['series'].str.replace('_pct', '').str.replace('_', ' ')
        long_B['series'] = long_B['series'].str.replace('Theoretical IPOF', 'Theoretical IPOF*').str.replace('ONLINE MO', 'ONLINE MO**')
        long_B.to_csv(csv_path_B, index=False)
        print(f"Saved CSV B to: {os.path.abspath(csv_path_B)}")

    plot_panels(
        dfA=dfA,
        dfB=dfB,
        out_path=out_path,
        panel=args.panel,
        group_fixed=args.group_fixed,
        lambda_fixed=args.lambda_fixed,
        ylim=args.ylim,
        dpi=args.dpi,
        style_scale=args.style_scale,
        lambda_list=lambda_list,
        group_sizes=group_sizes,
    )
    print(f"Saved plot to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()