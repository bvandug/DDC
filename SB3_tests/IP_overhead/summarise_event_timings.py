#!/usr/bin/env python3
"""
Summarize Simulink env timings from a TensorBoard scalars CSV.

Usage:
  python summarize_env_timings.py tb_scalars_YYYYMMDD_HHMMSS.csv

It expects rows with columns: tag, step, wall_time, value
and timing tags that look like:
  env_timings/reset/...
  env_timings/step/...
  env_timings/eval_out_angle  (etc.)
"""

import argparse
import pandas as pd
from typing import Tuple

def pm(mean: float, std: float | None) -> str:
    if pd.isna(std):
        return f"{mean:.6f}"
    return f"{mean:.6f} ± {std:.6f}"

def summarize(df: pd.DataFrame, title: str) -> pd.DataFrame:
    if df.empty:
        print(f"\n## {title}\n(no entries)\n")
        return df
    agg = (
        df.groupby("tag")["value"]
          .agg(["count", "mean", "std"])
          .sort_values("mean", ascending=False)
          .reset_index()
    )
    agg["mean±std (s)"] = agg.apply(lambda r: pm(r["mean"], r["std"]), axis=1)
    # Pretty print
    print(f"\n## {title}")
    print(agg[["tag", "count", "mean±std (s)"]].to_string(index=False))
    return agg

def split_scopes(env_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (reset_df, step_df, other_df) by inspecting tag paths."""
    reset_mask = env_df["tag"].str.contains(r"/reset/", case=False, na=False)
    step_mask  = env_df["tag"].str.contains(r"/step/",  case=False, na=False)

    df_reset = env_df[reset_mask].copy()
    df_step  = env_df[step_mask].copy()
    df_other = env_df[~(reset_mask | step_mask)].copy()  # e.g., env_timings/eval_out_angle (if present)
    return df_reset, df_step, df_other

def rollups(df_scope: pd.DataFrame, scope_name: str) -> None:
    """
    Print nice roll-ups if present:
      total_* keys, and a compute vs wall vs python split.
    """
    if df_scope.empty:
        return

    # Pull latest values by tag (or average; both are fine as we print mean ± std above)
    pivot = (
        df_scope.groupby("tag")["value"]
                .agg(["mean", "std", "count"])
    )

    # Known roll-up keys
    # (These are what we logged earlier; change these lists if your tag names differ.)
    tot_key   = f"env_timings/{scope_name}/total_{'step' if scope_name=='step' else 'reset'}_time"
    wall_key  = f"env_timings/{scope_name}/total_matlab_wall"
    comp_key  = f"env_timings/{scope_name}/total_matlab_compute"
    py_key    = f"env_timings/{scope_name}/total_python_time"

    # Gracefully handle absent keys
    def safe(tag: str) -> float | None:
        return None if tag not in pivot.index else float(pivot.loc[tag, "mean"])

    total   = safe(tot_key)
    wall    = safe(wall_key)
    comp    = safe(comp_key)
    py_time = safe(py_key)

    # Print if any exist
    if any(x is not None for x in (total, wall, comp, py_time)):
        print(f"\n### {scope_name.capitalize()} roll-ups")
        if total is not None:
            print(f"  • total   : {total:.6f} s")
        if wall is not None:
            print(f"  • matlab wall (IPC+compute): {wall:.6f} s")
        if comp is not None:
            print(f"  • matlab compute (tic/toc) : {comp:.6f} s")
        if (wall is not None) and (comp is not None):
            print(f"      ↳ IPC overhead (wall - compute): {(wall - comp):.6f} s")
        if py_time is not None:
            print(f"  • python-only               : {py_time:.6f} s")

def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("csv",default="training_timings/tb_scalars_20251007_140657.csv", help="TensorBoard scalars CSV exported from your run")
    # args = ap.parse_args()

    df = pd.read_csv("training_timings/tb_scalars_a2c.csv")

    # Focus on env timing tags only
    env_df = df[df["tag"].str.startswith("env_timings/", na=False)].copy()

    # Basic sanity
    if env_df.empty:
        print("No env_timings/* tags found in the CSV.")
        return

    # Split by scope
    df_reset, df_step, df_other = split_scopes(env_df)

    # Print summaries
    summarize(df_reset, "Reset timings (per tag)")
    rollups(df_reset, "reset")

    summarize(df_step, "Step timings (per tag)")
    rollups(df_step, "step")

    # "Overall" = all env_timings/* together (reset + step + other)
    summarize(env_df, "Overall timings (all env_timings/*)")

    # Optional: show 'other' block if it exists
    if not df_other.empty:
        summarize(df_other, "Other env timings (neither reset nor step)")

if __name__ == "__main__":
    main()
