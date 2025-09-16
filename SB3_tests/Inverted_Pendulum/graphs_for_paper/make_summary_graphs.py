#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_data(xlsx_path: Path) -> pd.DataFrame:
    # Read the "All" sheet by default; fall back to first sheet
    try:
        df = pd.read_excel(xlsx_path, sheet_name="All")
    except Exception:
        df = pd.read_excel(xlsx_path)  # first sheet
    # Clean types
    num_cols = [
        "env_noise",
        "run_noise",
        "mean_reward",
        "mean_stabilisation_time_s",
        "mean_steady_state_offset_deg",
        "mean_total_stable_time_s",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["algorithm"] = df["algorithm"].astype(str)
    return df.dropna(subset=["algorithm"])

def plot_noise_sensitivity(df: pd.DataFrame, outdir: Path):
    ensure_dir(outdir)
    # 1) Reward vs ENV noise (hold run_noise==0)
    fig, ax = plt.subplots()
    for algo, g in df[df["run_noise"] == 0.0].groupby("algorithm"):
        gg = g.sort_values("env_noise")
        if gg.empty:
            continue
        ax.plot(gg["env_noise"], gg["mean_reward"], marker="o", label=algo)
    ax.set_title("Mean Reward vs ENV Noise (run_noise = 0)")
    ax.set_xlabel("env_noise")
    ax.set_ylabel("mean_reward")
    ax.grid(True, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "reward_vs_env_noise_lines.png", dpi=200)
    plt.close(fig)

    # 2) Reward vs RUN noise (hold env_noise==0)
    fig, ax = plt.subplots()
    for algo, g in df[df["env_noise"] == 0.0].groupby("algorithm"):
        gg = g.sort_values("run_noise")
        if gg.empty:
            continue
        ax.plot(gg["run_noise"], gg["mean_reward"], marker="o", label=algo)
    ax.set_title("Mean Reward vs RUN Noise (env_noise = 0)")
    ax.set_xlabel("run_noise")
    ax.set_ylabel("mean_reward")
    ax.grid(True, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "reward_vs_run_noise_lines.png", dpi=200)
    plt.close(fig)

def plot_tradeoff_scatter(df: pd.DataFrame, outdir: Path):
    ensure_dir(outdir)
    # Reward vs Stabilisation time; size by total noise; color by algorithm (matplotlib default cycle)
    total_noise = df["env_noise"].fillna(0) + df["run_noise"].fillna(0)
    denom = (total_noise.max() if total_noise.max() and total_noise.max() > 0 else 1.0)
    size = 50 + 250 * (total_noise / denom)
    fig, ax = plt.subplots()
    for algo, g in df.groupby("algorithm"):
        ax.scatter(
            g["mean_stabilisation_time_s"],
            g["mean_reward"],
            s=size.loc[g.index],
            label=algo,
            alpha=0.8,
            edgecolors="none",
        )
    ax.set_title("Trade-off: Reward vs Stabilisation Time (marker size ~ total noise)")
    ax.set_xlabel("mean_stabilisation_time_s")
    ax.set_ylabel("mean_reward")
    ax.grid(True, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "tradeoff_reward_vs_stabtime.png", dpi=200)
    plt.close(fig)

def plot_offset_vs_noise(df: pd.DataFrame, outdir: Path):
    ensure_dir(outdir)
    # X: env_noise ; Y: steady-state offset ; marker size ~ run_noise
    fig, ax = plt.subplots()
    run_max = df["run_noise"].max()
    denom = (run_max if pd.notna(run_max) and run_max > 0 else 1.0)
    for algo, g in df.groupby("algorithm"):
        sizes = 50 + 250 * (g["run_noise"].fillna(0) / denom)
        ax.scatter(
            g["env_noise"],
            g["mean_steady_state_offset_deg"],
            s=sizes,
            label=algo,
            alpha=0.8,
            edgecolors="none",
        )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Steady-state Offset vs ENV Noise (marker size ~ run_noise)")
    ax.set_xlabel("env_noise")
    ax.set_ylabel("mean_steady_state_offset_deg")
    ax.grid(True, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "offset_vs_env_noise.png", dpi=200)
    plt.close(fig)

def plot_heatmaps(df: pd.DataFrame, outdir: Path):
    # One heatmap per algorithm: reward over env_noise × run_noise
    ensure_dir(outdir)
    env_vals = np.sort(df["env_noise"].unique())
    run_vals = np.sort(df["run_noise"].unique())

    for algo, g in df.groupby("algorithm"):
        pivot = g.pivot_table(
            index="run_noise", columns="env_noise",
            values="mean_reward", aggfunc="mean"
        ).reindex(index=run_vals, columns=env_vals)

        fig, ax = plt.subplots()
        im = ax.imshow(pivot.values, aspect="auto", origin="lower")
        ax.set_xticks(np.arange(len(env_vals)))
        ax.set_xticklabels([str(v) for v in env_vals], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(run_vals)))
        ax.set_yticklabels([str(v) for v in run_vals])
        ax.set_xlabel("env_noise")
        ax.set_ylabel("run_noise")
        ax.set_title(f"{algo}: Mean Reward heatmap")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("mean_reward")
        fig.tight_layout()
        fig.savefig(outdir / f"heatmap_reward_{algo}.png", dpi=200)
        plt.close(fig)

# --- Box & whisker plots (individual figures) --------------------------------
def plot_boxplots_by_noise(df: pd.DataFrame, outdir: Path):
    """
    For each metric, create two boxplots:
      1) distribution vs env_noise (all rows at each env_noise)
      2) distribution vs run_noise (all rows at each run_noise)
    Saves one PNG per figure, with legend for Median (orange) and Mean (green).
    """
    ensure_dir(outdir)

    metrics = [
        ("mean_reward", "Mean Reward"),
        ("mean_stabilisation_time_s", "Mean Stabilisation Time (s)"),
        ("mean_steady_state_offset_deg", "Mean Steady-State Offset (°)"),
        ("mean_total_stable_time_s", "Mean Total Stable Time (s)"),
    ]

    def _boxplot(df_grp_key: str, metric_key: str, metric_label: str, title_suffix: str, fname: str):
        # Sorted unique noise levels
        levels = np.sort(df[df_grp_key].dropna().unique())
        if len(levels) == 0:
            return

        # Collect arrays per level (drop NaNs)
        data = [df.loc[df[df_grp_key] == lv, metric_key].dropna().values for lv in levels]
        # Skip if everything is empty
        if all(len(arr) == 0 for arr in data):
            return

        fig, ax = plt.subplots()
        bp = ax.boxplot(
            data,
            labels=[str(lv) for lv in levels],
            showmeans=True,
            meanline=True,
            patch_artist=True,
        )
        # Light styling
        for patch in bp['boxes']:
            patch.set_alpha(0.5)

        # Legend for median (orange) and mean (green)
        median_line = bp['medians'][0] if bp['medians'] else None
        mean_line = bp['means'][0] if bp.get('means') else None
        handles = []
        if median_line is not None:
            handles.append(Line2D([0], [0], color=median_line.get_color(), linewidth=median_line.get_linewidth(), label="Median"))
        if mean_line is not None:
            handles.append(Line2D([0], [0], color=mean_line.get_color(), linewidth=mean_line.get_linewidth(), label="Mean"))
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8)

        ax.set_title(f"{metric_label} vs {title_suffix}")
        ax.set_xlabel(df_grp_key)
        ax.set_ylabel(metric_label)
        ax.grid(True, linestyle=":", axis="y")
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=200)
        plt.close(fig)

    for metric_key, metric_label in metrics:
        _boxplot(
            df_grp_key="env_noise",
            metric_key=metric_key,
            metric_label=metric_label,
            title_suffix="ENV Noise",
            fname=f"box_{metric_key}_by_env_noise.png",
        )
        _boxplot(
            df_grp_key="run_noise",
            metric_key=metric_key,
            metric_label=metric_label,
            title_suffix="RUN Noise",
            fname=f"box_{metric_key}_by_run_noise.png",
        )

# --- Single diagram with all boxplots (standardized y-lims per metric) --------
def plot_boxgrid_all_metrics(df: pd.DataFrame, outdir: Path):
    """
    Create ONE figure containing 4x2 subplots:
      rows   = metrics (Reward, Stab Time, Steady-State Offset, Total Stable Time)
      cols   = {ENV Noise, RUN Noise}
    Each subplot includes a legend for Median (orange) and Mean (green).
    For each metric, both columns share the same y-limits (computed from the whole df).
    """
    ensure_dir(outdir)

    metrics = [
        ("mean_reward", "Mean Reward"),
        ("mean_stabilisation_time_s", "Mean Stabilisation Time (s)"),
        ("mean_steady_state_offset_deg", "Mean Steady-State Offset (°)"),
        ("mean_total_stable_time_s", "Mean Total Stable Time (s)"),
    ]
    groupings = [("env_noise", "ENV Noise"), ("run_noise", "RUN Noise")]

    # Compute standardized y-limits per metric across the entire dataset
    ylims = {}
    for mkey, _ in metrics:
        s = pd.to_numeric(df[mkey], errors="coerce").dropna()
        if s.empty:
            ylims[mkey] = (0.0, 1.0)
        else:
            vmin = float(np.nanmin(s))
            vmax = float(np.nanmax(s))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = 0.0 if not np.isfinite(vmin) else vmin
                vmax = 1.0 if not np.isfinite(vmax) else vmax
                if vmin == vmax:
                    vmax = vmin + 1e-6
            pad = 0.05 * (vmax - vmin)
            ylims[mkey] = (vmin - pad, vmax + pad)

    # Build the grid
    nrows, ncols = len(metrics), len(groupings)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 14))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for r, (mkey, mlabel) in enumerate(metrics):
        for c, (gkey, gtitle) in enumerate(groupings):
            ax = axes[r, c]
            levels = np.sort(df[gkey].dropna().unique())
            if len(levels) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_axis_off()
                continue

            data = [df.loc[df[gkey] == lv, mkey].dropna().values for lv in levels]
            if all(len(arr) == 0 for arr in data):
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_axis_off()
                continue

            bp = ax.boxplot(
                data,
                labels=[str(lv) for lv in levels],
                showmeans=True,
                meanline=True,
                patch_artist=True,
            )
            for patch in bp['boxes']:
                patch.set_alpha(0.5)

            # Standardized y-limits for this metric across both columns
            ax.set_ylim(*ylims[mkey])

            # Legends: Median (orange) and Mean (green)
            median_line = bp['medians'][0] if bp['medians'] else None
            mean_line = bp['means'][0] if bp.get('means') else None
            handles = []
            if median_line is not None:
                handles.append(Line2D([0], [0], color=median_line.get_color(), linewidth=median_line.get_linewidth(), label="Median"))
            if mean_line is not None:
                handles.append(Line2D([0], [0], color=mean_line.get_color(), linewidth=mean_line.get_linewidth(), label="Mean"))
            if handles:
                ax.legend(handles=handles, loc="upper right", fontsize=8)

            if r == 0:
                ax.set_title(gtitle)
            if c == 0:
                ax.set_ylabel(mlabel)
            ax.set_xlabel(gkey)
            ax.grid(True, linestyle=":", axis="y")
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")

    fig.tight_layout()
    fig.savefig(outdir / "boxplots_all_metrics_env_and_run.png", dpi=200)
    plt.close(fig)

def compute_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustness to ENV noise (run_noise == 0):
      reward(env_noise = max, run=0) / reward(env_noise = 0, run=0)
    Robustness to RUN noise (env_noise == 0):
      reward(run_noise = max, env=0) / reward(run_noise = 0, env=0)
    """
    rows = []
    for algo, g in df.groupby("algorithm"):
        g0 = g[(g["env_noise"] == 0.0) & (g["run_noise"] == 0.0)]
        base = g0["mean_reward"].mean() if not g0.empty else np.nan

        # env robustness
        env_max = g[g["run_noise"] == 0.0]
        if not env_max.empty:
            emax_val = env_max.loc[env_max["env_noise"].idxmax(), "mean_reward"]
        else:
            emax_val = np.nan
        env_rob = emax_val / base if (base and not np.isnan(base)) else np.nan

        # run robustness
        run_max = g[g["env_noise"] == 0.0]
        if not run_max.empty:
            rmax_val = run_max.loc[run_max["run_noise"].idxmax(), "mean_reward"]
        else:
            rmax_val = np.nan
        run_rob = rmax_val / base if (base and not np.isnan(base)) else np.nan

        rows.append({
            "algorithm": algo,
            "base_reward_env0_run0": base,
            "reward_env_max_run0": emax_val,
            "robustness_env_noise": env_rob,
            "reward_run_max_env0": rmax_val,
            "robustness_run_noise": run_rob,
        })
    return pd.DataFrame(rows)

def plot_robustness_bars(rob: pd.DataFrame, outdir: Path):
    ensure_dir(outdir)
    # Two bar charts: env robustness and run robustness
    for col, fname, title in [
        ("robustness_env_noise", "robustness_env_noise.png", "Robustness to ENV noise (higher is better)"),
        ("robustness_run_noise", "robustness_run_noise.png", "Robustness to RUN noise (higher is better)"),
    ]:
        r = rob.sort_values(col, ascending=False)
        fig, ax = plt.subplots()
        ax.bar(r["algorithm"], r[col])
        ax.axhline(1.0, linestyle="--", linewidth=1)  # baseline (no drop)
        ax.set_title(title)
        ax.set_ylabel("ratio vs. noise-free reward")
        ax.set_ylim(bottom=0)
        for i, v in enumerate(r[col].values):
            if pd.notna(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, rotation=0)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=200)
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Generate graphs from summary_metrics.xlsx")
    ap.add_argument("--xlsx", type=Path, default=Path("summary_metrics.xlsx"))
    ap.add_argument("--outdir", type=Path, default=Path("analysis_graphs"))
    args = ap.parse_args()

    df = load_data(args.xlsx)
    ensure_dir(args.outdir)

    # Plots
    plot_noise_sensitivity(df, args.outdir)
    plot_tradeoff_scatter(df, args.outdir)
    plot_offset_vs_noise(df, args.outdir)
    plot_heatmaps(df, args.outdir)

    # Box & whisker plots (individual figures per metric + grouping)
    plot_boxplots_by_noise(df, args.outdir)

    # Single diagram with all metrics & standardized y-lims (with mean/median legends)
    plot_boxgrid_all_metrics(df, args.outdir)

    # Robustness
    rob = compute_robustness(df)
    rob.to_csv(args.outdir / "robustness_summary.csv", index=False)
    plot_robustness_bars(rob, args.outdir)

    print(f"Saved graphs & robustness CSV to: {args.outdir}")

if __name__ == "__main__":
    main()
