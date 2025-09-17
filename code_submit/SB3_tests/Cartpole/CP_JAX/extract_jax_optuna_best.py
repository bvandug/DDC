import argparse
import json
import os
from pathlib import Path
from typing import Optional, List
import optuna

ALGOS = ["a2c", "ddpg", "dqn", "ppo", "sac", "td3"]


def storage_url_for_sqlite(path: Path) -> str:
    """Build an Optuna-compatible SQLite storage URL from a filesystem path.

        Optuna expects a URL of the form ``sqlite:///absolute/path`` with
        forward slashes, even on Windows. This helper resolves the path and
        normalizes separators.

        Parameters
        ----------
        path : pathlib.Path
            Path to a ``.db`` file.

        Returns
        -------
        str
            Storage URL suitable for Optuna's ``storage=`` argument.
    """

    abs_path = path.resolve()
    return f"sqlite:///{str(abs_path).replace(os.sep, '/')}"
    

def find_db_for_algo(db_dir: Path, algo: str) -> Optional[Path]:
    """Locate an Optuna SQLite DB for a given algorithm under a directory.

        Search order:
        1) Exact filenames in ``db_dir``:
        - ``ip_jax_optuna_{algo}.db``
        - ``jax_optuna_{algo}.db``
        2) Recursive fallback: any ``*.db`` whose name contains both
        ``optuna`` and the ``algo`` substring (case-insensitive).

        Parameters
        ----------
        db_dir : pathlib.Path
            Directory to search.
        algo : str
            Algorithm key (e.g., ``"ppo"``, ``"sac"``).

        Returns
        -------
        pathlib.Path | None
            Path to the first matching DB, or ``None`` if not found.
    """

    candidates = [
        db_dir / f"ip_jax_optuna_{algo}.db",
        db_dir / f"jax_optuna_{algo}.db",
    ]
    for c in candidates:
        if c.exists():
            return c

    for p in db_dir.rglob("*.db"):
        name = p.name.lower()
        if "optuna" in name and algo in name:
            return p

    return None


def choose_study_name(storage_url: str, algo: str) -> Optional[str]:
    """Choose the most appropriate study name within a storage for an algo.

        Heuristic:
        1) Prefer exact matches in this order:
        ``"jax_{algo}_tuning"``, ``"ip_jax_{algo}_tuning"``, ``"{algo}_tuning"``.
        2) Otherwise, pick the study with the highest ``best_value``. If a
        study's direction is MINIMIZE, its value is sign-flipped so the
        comparison remains meaningful.

        Parameters
        ----------
        storage_url : str
            Optuna storage URL (e.g., from ``storage_url_for_sqlite``).
        algo : str
            Algorithm key used in preferred-name matching.

        Returns
        -------
        str | None
            Selected study name, or ``None`` if no studies are present.
    """

    summaries: List[optuna.study.StudySummary] = optuna.study.get_all_study_summaries(storage=storage_url)
    if not summaries:
        return None

    preferred = [f"jax_{algo}_tuning", f"ip_jax_{algo}_tuning", f"{algo}_tuning"]
    for pref in preferred:
        for s in summaries:
            if s.study_name == pref:
                return s.study_name

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
    """Extract best trial info from a DB and write a per-algo JSON file.

        Loads the best trial from the chosen study, determines the total number
        of trials (via study summaries when available), and writes a JSON file
        containing:
        ``{"best_params": {...}, "best_value": float|None, "n_trials": int}``.

        Parameters
        ----------
        db_path : pathlib.Path
            Path to the Optuna SQLite database.
        algo : str
            Algorithm key used to name the output file.
        out_dir : pathlib.Path
            Directory where the JSON will be written (created if missing).

        Returns
        -------
        pathlib.Path | None
            The written JSON filepath, or ``None`` if no study could be chosen.

        Raises
        ------
        optuna.exceptions.OptunaError
            If loading the study fails for other reasons.
    """

    
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
    """CLI entry point: scan DBs, extract best trials, and write JSON files.

        Parses arguments:
        - ``--db-dir``: directory containing the ``*.db`` files (default: ``.``).
        - ``--out-dir``: directory for output JSON files (default: ``final_jax_results``).
        - ``--algos``: comma-separated list of algorithm keys to process
        (default: ``a2c,ddpg,dqn,ppo,sac,td3``).

        For each requested algorithm, attempts to locate a matching Optuna DB,
        loads the most suitable study, and writes ``{algo}_best_params.json``.
        Prints warnings for missing DBs and a summary at the end.

        Returns
        -------
        None
    """

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
