# src/idle_ratio_ipact_offline_fig5_bars.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (imported for 3D projection)
from matplotlib import cm
from matplotlib.colors import Normalize

from src.common import ensure_dir


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_grid_df(
    lambda_os_list: list[float],
    group_sizes: list[int],
    panel_scales: list[float],
    runs: int,
    seed: int,
    N: int,
    guard_ns: float,
    dmin_km: float,
    dmax_km: float,
) -> pd.DataFrame:
    rows = []
    Guard = guard_ns * 1e-9  # to seconds
    T_max = 1.0 / 1000.0  # fixed 1 ms
    panel_letters = {0.3: 'a', 0.6: 'b', 1.0: 'c'}
    panel_indices = {s: idx for idx, s in enumerate(sorted(panel_scales))}
    combos = [(g, lam) for g in group_sizes for lam in lambda_os_list]
    combo_dict = {combo: idx for idx, combo in enumerate(combos)}
    for s in panel_scales:
        panel = panel_letters[s]
        p_idx = panel_indices[s]
        for g in group_sizes:
            for lam in lambda_os_list:
                combo = (g, lam)
                c_idx = combo_dict[combo]
                idle_ratios = []
                for run in range(runs):
                    run_seed = seed + p_idx * 10000 + c_idx * 100 + run
                    rng = np.random.default_rng(run_seed)
                    dists = rng.uniform(dmin_km, dmax_km, N)
                    taus = dists * 5e-6  # seconds
                    indices = rng.choice(N, g, replace=False)
                    group_taus = taus[indices]
                    min_tau = np.min(group_taus)
                    max_tau = np.max(group_taus)
                    idle_fixed = min_tau + max_tau - Guard
                    if idle_fixed < 0:
                        idle_fixed = 0
                    guard_total = (g - 1) * Guard
                    rho_eff = s * lam * (g / N)
                    send_time = rho_eff * T_max
                    cycle_len = idle_fixed + guard_total + send_time
                    if cycle_len < 1e-9:
                        cycle_len = 1e-9
                    idle_ratio = idle_fixed / cycle_len
                    idle_ratios.append(idle_ratio)
                mean = np.mean(idle_ratios)
                std = np.std(idle_ratios, ddof=1) if runs > 1 else 0.0
                pct = 100.0 * mean
                rows.append(
                    {
                        "panel": panel,
                        "lambda_os": lam,
                        "group_size": g,
                        "idle_ratio_mean": float(mean),
                        "idle_ratio_std": float(std),
                        "idle_ratio_pct": float(pct),
                    }
                )
                print(f"panel={panel} s={s:.1f} |Os|={g} lambda_os={lam:.1f} -> idle={pct:.2f}%", flush=True)
    return pd.DataFrame(rows)


def plot_3panels(
    df: pd.DataFrame,
    out_path: str,
    elev: float,
    azim: float,
    dist: float,
    box_aspect: tuple[float, float, float] | None,
    zmax: float,
    dpi: int,
) -> None:
    # Filter panels
    panel_dfs = {
        'a': df[df['panel'] == 'a'],
        'b': df[df['panel'] == 'b'],
        'c': df[df['panel'] == 'c'],
    }

    # Unique values
    rho_vals = sorted(df['lambda_os'].unique())
    grp_vals = sorted(df['group_size'].unique())
    nt = len(rho_vals)
    ng = len(grp_vals)

    # Grid for bar positions
    _xx, _yy = np.meshgrid(range(nt), range(ng))
    x = _xx.ravel()
    y = _yy.ravel()
    dx = dy = 0.8

    # Global min/max for color scale
    all_pct = df['idle_ratio_pct'].values
    pct_min = all_pct.min()
    pct_max = all_pct.max()

    # Save meta CSV
    meta_path = out_path.replace('.png', '_meta.csv')
    meta_df = pd.DataFrame({'min_pct': [pct_min], 'max_pct': [pct_max]})
    meta_df.to_csv(meta_path, index=False)

    # Determine z_max
    z_max = zmax if pct_max <= zmax else np.ceil(pct_max / 5) * 5

    # Shared colormap
    norm = Normalize(vmin=pct_min, vmax=pct_max)
    cmap = cm.viridis

    # Figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

    panel_titles = {
        'a': r'(a) $\lambda_{OT} = 0.3$',
        'b': r'(b) $\lambda_{OT} = 0.6$',
        'c': r'(c) $\lambda_{OT} = 1.0$',
    }

    for i, p in enumerate(['a', 'b', 'c']):
        ax = axs[i]
        subdf = panel_dfs[p]

        # Collect top values in row-major order
        top_vals: list[float] = []
        for gy in range(ng):
            for gx in range(nt):
                v = subdf[(subdf['lambda_os'] == rho_vals[gx]) & (subdf['group_size'] == grp_vals[gy])]['idle_ratio_pct'].iloc[0]
                top_vals.append(float(v))
        top = np.array(top_vals)

        colors = cmap(norm(top))

        ax.bar3d(x, y, 0, dx, dy, top, shade=True, color=colors, edgecolor='black', linewidth=0.2)

        ax.set_zlim(0, z_max)

        ax.set_xticks(range(nt))
        ax.set_xticklabels([f"{r:.1f}" for r in rho_vals])
        ax.set_xlabel(r'$\lambda_{os}$')

        ax.set_yticks(range(ng))
        ax.set_yticklabels([str(g) for g in grp_vals])
        ax.set_ylabel(r'$|O_s|$')

        ax.set_zlabel("Average idle ratio [%]")

        ax.set_title(panel_titles[p])

        ax.view_init(elev=elev, azim=azim)
        ax.dist = dist

        if box_aspect is not None:
            ax.set_box_aspect(box_aspect)

    fig.suptitle("Figure 5 – IPACT (Offline): Average idle ratio vs load and group size")

    # Shared colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), orientation='horizontal', pad=0.07, fraction=0.05, shrink=0.8)
    cbar.set_label("Average idle ratio [%]")
    cbar.set_ticks(np.arange(0, pct_max + 5, 5))

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate bar plots for Figure 5 – IPACT (Offline)")
    parser.add_argument("--lambda_os_list", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--group_sizes", type=str, default="2,4,8,16,24,32")
    parser.add_argument("--panel_scales", type=str, default="0.3,0.6,1.0")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--cycles", type=int, default=3500)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=str, default="out/fig5_idle_ratio_ipact_offline")
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--Rb", type=float, default=10e9)
    parser.add_argument("--guard_ns", type=float, default=624)
    parser.add_argument("--dmin_km", type=float, default=10.0)
    parser.add_argument("--dmax_km", type=float, default=20.0)
    parser.add_argument("--elev", type=float, default=22)
    parser.add_argument("--azim", type=float, default=-130)
    parser.add_argument("--dist", type=float, default=9.5)
    parser.add_argument("--box_aspect", type=str, default="1,1,0.7")
    parser.add_argument("--zmax", type=float, default=30)
    parser.add_argument("--dpi", type=int, default=160)

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    lambda_os_list = parse_float_list(args.lambda_os_list)
    group_sizes = parse_int_list(args.group_sizes)
    panel_scales = parse_float_list(args.panel_scales)

    df = build_grid_df(
        lambda_os_list=lambda_os_list,
        group_sizes=group_sizes,
        panel_scales=panel_scales,
        runs=args.runs,
        seed=args.seed,
        N=args.N,
        guard_ns=args.guard_ns,
        dmin_km=args.dmin_km,
        dmax_km=args.dmax_km,
    )
    csv_path = os.path.join(args.out_dir, "fig5_grid.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {os.path.abspath(csv_path)}")

    out_path = os.path.join(args.out_dir, "fig5_bars_3panels.png")
    box_aspect_tuple = tuple(parse_float_list(args.box_aspect)) if args.box_aspect else None
    plot_3panels(
        df=df,
        out_path=out_path,
        elev=args.elev,
        azim=args.azim,
        dist=args.dist,
        box_aspect=box_aspect_tuple,
        zmax=args.zmax,
        dpi=args.dpi,
    )
    print(f"Saved plot to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
