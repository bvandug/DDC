#!/usr/bin/env python3
"""
Reorganize RL checkpoint folders to the canonical layout:

<dst>/<algo>_noise_<level>/
    best_model.zip
    jax_models/
        <all other .zip checkpoints>
        <replay buffers / misc artifacts>

Works with structures like:
- jax/noise_0.000_a2c/jax/noise_0.000_a2c/noise_0.000_best_model_*.zip
- noise_0.000_ddpg/jax/noise_0.000_best_model[.zip]
- noise_0.000_ddpg/jax/noise_0.000_best_model_replay_buffer[.pkl]

USAGE:
  python reorganize_models.py --src /path/to/root [--dst /path/to/out] [--apply]

By default it's a dry-run. Use --apply to actually move/rename files.
"""

import argparse
import re
import shutil
from pathlib import Path

ALGOS = {"a2c", "ddpg", "td3", "ppo", "sac", "dqn"}  # extend if needed

BEST_CAND_RE = re.compile(r"""
    ^
    (?P<prefix>.*?)
    (best[_-]?model)            # 'best_model' variants
    (?P<step>[_-]?\d+)?         # optional _25000 etc
    (?P<ext>\.zip|\.7z|\.tar\.gz|\.tgz|\.tar|\.pkl|\.bin)?  # optional ext
    $
""", re.VERBOSE | re.IGNORECASE)

NOISE_DIR_RE = re.compile(r"^noise_(?P<noise>\d+\.\d{3})_(?P<algo>[a-z0-9_]+)$", re.IGNORECASE)

def sniff_zip(path: Path) -> bool:
    """Return True if file begins with ZIP local file header 'PK\\x03\\x04'."""
    try:
        with path.open("rb") as f:
            sig = f.read(4)
        return sig == b"PK\x03\x04"
    except Exception:
        return False

def ensure_dir(p: Path, dry: bool):
    if dry:
        print(f"[DRY] mkdir -p {p}")
        return
    p.mkdir(parents=True, exist_ok=True)

def move(src: Path, dst: Path, dry: bool):
    ensure_dir(dst.parent, dry)
    if dry:
        print(f"[DRY] mv {src} -> {dst}")
    else:
        shutil.move(str(src), str(dst))

def copy(src: Path, dst: Path, dry: bool):
    ensure_dir(dst.parent, dry)
    if dry:
        print(f"[DRY] cp {src} -> {dst}")
    else:
        shutil.copy2(str(src), str(dst))

def choose_best_zip(candidates):
    """
    From a list of files named like ...best_model_<steps>.zip or best_model.zip,
    prefer plain 'best_model.zip'; else highest step number.
    """
    if not candidates:
        return None
    # Prefer exact best_model.zip
    for c in candidates:
        if c.name.lower().endswith("best_model.zip") or re.search(r"best[_-]?model\.zip$", c.name, re.I):
            return c
    # Else highest step number
    def step_num(p: Path):
        m = BEST_CAND_RE.match(p.name)
        if m and m.group("step"):
            return int(re.sub(r"[^\d]", "", m.group("step")))
        return -1
    return max(candidates, key=step_num)

def parse_noise_algo_from_dirname(name: str):
    """
    Try direct pattern 'noise_0.000_a2c' first; otherwise try to infer from inside path parts.
    """
    m = NOISE_DIR_RE.match(name)
    if m:
        noise = m.group("noise")
        algo = m.group("algo").lower()
        # normalize algo (strip trailing underscores)
        algo = algo.rstrip("_")
        return algo, noise
    # Try looser inference
    # Find algo token
    algo = None
    for a in ALGOS:
        if re.search(fr"(?:^|[_-]){a}(?:[_-]|$)", name, re.I):
            algo = a
            break
    # Find noise token like 0.000 or 0p000
    noise = None
    m2 = re.search(r"(\d+\.\d{3})", name)
    if m2:
        noise = m2.group(1)
    return algo, noise

def collect_candidates(run_dir: Path):
    """
    Walk a run directory and collect:
      - all .zip files
      - a single 'best' file if present (zip or not)
      - other artifacts (e.g., replay buffers)
    """
    zips = []
    best_like = []
    others = []

    for p in run_dir.rglob("*"):
        if p.is_dir():
            continue
        name = p.name.lower()
        if name.endswith(".zip"):
            zips.append(p)
            if "best" in name:
                best_like.append(p)
        else:
            if "best" in name:
                best_like.append(p)
            else:
                # keep replay buffers and misc
                others.append(p)

    return zips, best_like, others

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root folder to scan (contains 'jax/', 'noise_*/', etc.)")
    ap.add_argument("--dst", default=None, help="Destination root (default: same as --src)")
    ap.add_argument("--apply", action="store_true", help="Actually move/rename files (otherwise dry-run)")
    args = ap.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve() if args.dst else src_root
    dry = not args.apply

    print(f"Scanning: {src_root}")
    print(f"Output to: {dst_root}")
    print(f"Mode: {'DRY-RUN' if dry else 'APPLY'}")

    # Heuristic: candidates are immediate subdirs under src_root and also nested 'jax/noise_*_*' leaves
    run_dirs = []

    # (1) Direct children like noise_0.000_ddpg
    for d in src_root.iterdir():
        if d.is_dir() and d.name.lower().startswith("noise_"):
            run_dirs.append(d)

    # (2) Nested under jax/noise_.../jax/noise_... (your second screenshot)
    jax_dir = src_root / "jax"
    if jax_dir.is_dir():
        for d in jax_dir.rglob("noise_*_*"):
            if d.is_dir():
                run_dirs.append(d)

    # De-duplicate
    seen = set()
    unique_run_dirs = []
    for d in run_dirs:
        if d.resolve() not in seen:
            unique_run_dirs.append(d)
            seen.add(d.resolve())

    if not unique_run_dirs:
        print("No run directories found. Nothing to do.")
        return

    for run in unique_run_dirs:
        algo, noise = parse_noise_algo_from_dirname(run.name)
        # Fall back to parsing parent names if needed
        if (algo is None or noise is None):
            algo2, noise2 = parse_noise_algo_from_dirname(run.parent.name)
            if algo is None: algo = algo2
            if noise is None: noise = noise2

        if algo not in ALGOS or noise is None:
            print(f"[SKIP] Could not infer algo/noise from '{run}'.")
            continue

        dst_dir = dst_root / f"{algo}_noise_{noise}"
        jax_models_dir = dst_dir / "jax_models"

        ensure_dir(jax_models_dir, dry)

        zips, best_like, others = collect_candidates(run)

        # Choose best zip; if none, try to upgrade a best-like file without extension to .zip if it is zip
        best_zip = choose_best_zip([p for p in best_like if p.suffix.lower() == ".zip"])

        if not best_zip and best_like:
            # check if first best_like is actually a zip with missing extension
            for cand in best_like:
                if cand.is_file() and sniff_zip(cand):
                    # treat as zip
                    best_zip = cand
                    break

        # Move best model
        if best_zip:
            dst_best = dst_dir / "best_model.zip"
            if best_zip.suffix.lower() != ".zip":
                # add .zip extension if missing
                dst_best = dst_dir / "best_model.zip"
            if best_zip.resolve() == dst_best.resolve():
                print(f"[INFO] best_model.zip already in place for {dst_dir.name}")
            else:
                copy(best_zip, dst_best, dry)
        else:
            print(f"[WARN] No best_model.zip found/inferred for '{run}'.")
            # As a fallback, if any zips exist pick the largest (often latest)
            if zips:
                largest = max(zips, key=lambda p: p.stat().st_size)
                print(f"       -> Using largest zip as best: {largest.name}")
                copy(largest, dst_dir / "best_model.zip", dry)

        # Move remaining zips (excluding best we just copied)
        for z in zips:
            # Skip the one we used as best (by content path)
            if best_zip and z.resolve() == best_zip.resolve():
                continue
            move(z, jax_models_dir / z.name, dry)

        # Move other artifacts (replay buffers etc.) into jax_models
        for o in others:
            move(o, jax_models_dir / o.name, dry)

        print(f"[OK] Standardized -> {dst_dir}")

    print("Done.")

if __name__ == "__main__":
    main()
