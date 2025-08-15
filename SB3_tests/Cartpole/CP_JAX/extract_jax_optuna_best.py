#!/usr/bin/env python3
"""
Extract the top (best) trial from each Optuna SQLite DB and write per‑algo JSON
files like:
{
    "best_params": {...},
    "best_value": 498.77,
    "n_trials": 50
}

It looks for the 6 algos: a2c, ddpg, dqn, ppo, sac, td3
DB filename patterns handled (in --db-dir):
  - ip_jax_optuna_{algo}.db     (as used in ip_jax_hp.py)
  - jax_optuna_{algo}.db        (as in your screenshot)

Usage:
  python extract_jax_optuna_best.py --db-dir . --out-dir final_jax_results
"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional, List

import optuna


ALGOS = ["a2c", "ddpg", "dqn", "ppo", "sac", "td3"]


def storage_url_for_sqlite(path: Path) -> str:
    # Optuna expects "sqlite:///<abs_path>" with forward slashes
    abs_path = path.resolve()
    return f"sqlite:///{str(abs_path).replace(os.sep, '/')}"
    

def find_db_for_algo(db_dir: Path, algo: str) -> Optional[Path]:
    """Return path to DB for the algo, or None if not found."""
    candidates = [
        db_dir / f"ip_jax_optuna_{algo}.db",
        db_dir / f"jax_optuna_{algo}.db",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback: search recursively for something like "*optuna*{algo}*.db"
    for p in db_dir.rglob("*.db"):
        name = p.name.lower()
        if "optuna" in name and algo in name:
            return p

    return None


def choose_study_name(storage_url: str, algo: str) -> Optional[str]:
    """Pick the most sensible study in this DB for the given algo."""
    summaries: List[optuna.study.StudySummary] = optuna.study.get_all_study_summaries(storage=storage_url)
    if not summaries:
        return None

    preferred = [f"jax_{algo}_tuning", f"ip_jax_{algo}_tuning", f"{algo}_tuning"]
    for pref in preferred:
        for s in summaries:
            if s.study_name == pref:
                return s.study_name

    # Otherwise: take the study with the highest best_value (assuming maximize)
    # Note: some studies may have None best_value if no complete trials.
    def key_fn(s: optuna.study.StudySummary):
        # Fallback to -inf if best_value is None
        val = s.best_value if s.best_value is not None else float("-inf")
        try:
            # If direction is MINIMIZE, invert sign so max() works
            if getattr(s.direction, "name", "").upper().endswith("MINIMIZE"):
                val = -val
        except Exception:
            pass
        return val

    summaries_sorted = sorted(summaries, key=key_fn, reverse=True)
    return summaries_sorted[0].study_name if summaries_sorted else None


def extract_and_write(db_path: Path, algo: str, out_dir: Path) -> Optional[Path]:
    """Load best trial and write JSON. Returns written file path or None."""
    storage_url = storage_url_for_sqlite(db_path)
    study_name = choose_study_name(storage_url, algo)
    if study_name is None:
        print(f"[WARN] No studies found in DB: {db_path}")
        return None

    study = optuna.load_study(study_name=study_name, storage=storage_url)

    # Get n_trials via summaries to avoid counting only finished trials
    n_trials = None
    for s in optuna.study.get_all_study_summaries(storage=storage_url):
        if s.study_name == study_name:
            n_trials = s.n_trials
            break
    if n_trials is None:
        n_trials = len(study.trials)

    payload = {
        "best_params": study.best_params,
        "best_value": float(study.best_value) if study.best_value is not None else None,
        "n_trials": int(n_trials),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo}_best_params.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"[OK] {algo.upper():4s}: wrote {out_path} (from {db_path.name}, study='{study_name}')")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-dir", type=Path, default=Path("."), help="Directory that contains the *.db files")
    ap.add_argument("--out-dir", type=Path, default=Path("final_jax_results"), help="Where to write JSON files")
    ap.add_argument("--algos", type=str, default=",".join(ALGOS), help="Comma-separated algos (default: a2c,ddpg,dqn,ppo,sac,td3)")
    args = ap.parse_args()

    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    missing = []

    for algo in algos:
        db_path = find_db_for_algo(args.db_dir, algo)
        if not db_path:
            print(f"[WARN] Could not find DB for '{algo}' under {args.db_dir}. Expected one of:")
            print(f"       ip_jax_optuna_{algo}.db  or  jax_optuna_{algo}.db")
            missing.append(algo)
            continue
        try:
            extract_and_write(db_path, algo, args.out_dir)
        except Exception as e:
            print(f"[ERROR] Failed to extract for {algo} from {db_path}: {e}")

    if missing:
        print(f"\nDone with warnings. Missing algos: {', '.join(missing)}")
    else:
        print("\nAll requested algos processed.")


if __name__ == "__main__":
    main()
