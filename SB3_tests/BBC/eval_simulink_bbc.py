# eval_simulink_bbc.py - NP-style evaluation and metrics for Simulink env
import argparse
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import A2C, SAC, TD3, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from BBCSimulink_env import BBCSimulinkEnv, DiscretizeDutyWrapper

ALGO_MAP = {"a2c": A2C, "sac": SAC, "td3": TD3, "dqn": DQN}

# ------------------------- metrics & plotting -------------------------

def calculate_performance_metrics(voltages_arr, times_arr, target_voltage, tolerance):
    """
    Same metrics as NP evaluator:
      - stabilisation_time (first time it permanently enters ±tolerance)
      - steady_state_error (mean after stabilisation)
      - overshoot (max above target)
      - undershoot (max dip below target after first crossing)
    """
    if len(times_arr) == 0:
        return {
            "stabilisation_time": np.nan, "steady_state_error": np.nan,
            "overshoot": np.nan, "undershoot": np.nan
        }

    outside = np.where(np.abs(voltages_arr - target_voltage) > tolerance)[0]
    has_stabilized = False
    stabilisation_time = times_arr[-1]
    steady_state_error = np.nan

    if len(outside) == 0:
        first_stable_index = 0
        has_stabilized = True
        stabilisation_time = times_arr[0]
    else:
        last_unstable_index = outside[-1]
        if last_unstable_index + 1 < len(times_arr):
            first_stable_index = last_unstable_index + 1
            stabilisation_time = times_arr[first_stable_index]
            has_stabilized = True

    if has_stabilized:
        steady_state_voltages = voltages_arr[first_stable_index:]
        steady_state_error = float(np.mean(steady_state_voltages - target_voltage))

    overshoot = float(max(0.0, np.max(voltages_arr) - target_voltage))

    # undershoot = max dip below target after first time >= target
    first_cross_indices = np.where(voltages_arr >= target_voltage)[0]
    if len(first_cross_indices) > 0:
        first_cross_index = int(first_cross_indices[0])
        voltages_after_cross = voltages_arr[first_cross_index:]
        undershoot = float(max(0.0, target_voltage - np.min(voltages_after_cross)))
    else:
        undershoot = 0.0

    return {
        "stabilisation_time": float(stabilisation_time),
        "steady_state_error": float(steady_state_error),
        "overshoot": float(overshoot),
        "undershoot": float(undershoot)
    }


def plot_and_save_summary(all_episode_data, target_voltage, tolerance, model_type, out_dir):
    """
    Save raw CSV for all episodes and two plots (full + zoomed),
    mirroring the NP evaluator’s outputs.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Raw CSV (episode,time_s,voltage_v,duty_cycle)
    csv_path = os.path.join(out_dir, "all_episodes_trajectory.csv")
    rows = []
    for i, (times, voltages, duties) in enumerate(all_episode_data):
        ep = np.full_like(times, i + 1, dtype=int)
        rows.append(np.column_stack([ep, times, voltages, duties]))
    if rows:
        data = np.vstack(rows)
        header = "episode,time_s,voltage_v,duty_cycle"
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
        print(f"  - Saved CSV: {csv_path}")

    # 2) Full plot
    full_plot_path = os.path.join(out_dir, "response_plot_full.png")
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    for i, (times, voltages, _) in enumerate(all_episode_data):
        ax1.plot(times * 1000.0, voltages, label=f"Episode {i+1}", alpha=0.85)
    ax1.axhline(y=target_voltage, linestyle="--", label="Target", color="r")
    ax1.axhspan(target_voltage - tolerance, target_voltage + tolerance, alpha=0.1, color="r",
                label=f"±{tolerance} V")
    ax1.set_title(f"{model_type} Agent Performance - Full View")
    ax1.set_xlabel("Time (ms)"); ax1.set_ylabel("Output Voltage (V)")
    ax1.grid(True); ax1.legend()
    fig1.savefig(full_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # 3) Zoomed plot
    zoom_plot_path = os.path.join(out_dir, "response_plot_zoomed.png")
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    for i, (times, voltages, _) in enumerate(all_episode_data):
        ax2.plot(times * 1000.0, voltages, marker=".", linestyle="-", label=f"Episode {i+1}", alpha=0.9)
    ax2.axhline(y=target_voltage, linestyle="--", label="Target", color="r")
    ax2.axhspan(target_voltage - tolerance, target_voltage + tolerance, alpha=0.1, color="r")
    ax2.set_title(f"{model_type} Agent Performance - Zoomed View")
    ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Output Voltage (V)")
    ax2.set_ylim(target_voltage - (tolerance * 4), target_voltage + (tolerance * 4))
    ax2.grid(True); ax2.legend()
    fig2.savefig(zoom_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"  - Saved plots to: {out_dir}")


# ------------------------- core evaluation -------------------------

def make_simulink_env(simulink_model: str, algo: str, dqn_bins: int, voltage_noise_std: float):
    """
    Construct BBCSimulinkEnv with optional DQN discretization and noise parity.
    """
    def env_fn():
        env = BBCSimulinkEnv(model_name=simulink_model, voltage_noise_std=voltage_noise_std)
        if algo.lower() == "dqn":
            env = DiscretizeDutyWrapper(
                env,
                n_bins=dqn_bins,
                low=env.action_space.low[0],
                high=env.action_space.high[0],
            )
        return env
    return env_fn


def evaluate_single_model(model_path: str, stats_path: str, algo: str, simulink_model: str,
                          out_dir: str, num_episodes: int, dqn_bins: int,
                          voltage_noise_std: float, tolerance: float, k_macro: int = 1):
    """
    Evaluate one model for N episodes, save metrics/plots/CSV into out_dir.
    """
    if not os.path.exists(model_path):
        print(f"[WARN] Missing model: {model_path}"); return
    if not os.path.exists(stats_path):
        print(f"[WARN] Missing VecNormalize stats: {stats_path}"); return

    # Build env
    env_fn = make_simulink_env(simulink_model, algo, dqn_bins, voltage_noise_std)
    env = DummyVecEnv([env_fn])

    # Load normalization + model
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    algo_cls = ALGO_MAP.get(algo.lower())
    if algo_cls is None:
        print(f"[WARN] Unknown algo: {algo}"); env.close(); return
    model = algo_cls.load(model_path, env=env)
    print(f"  - Loaded model: {model_path}")

    # For time axis
    target_voltage = env.get_attr("target_voltage")[0]
    dt_sim = env.get_attr("dt")[0]
    frame_skip = env.get_attr("frame_skip")[0]
    dt_step = dt_sim * frame_skip * max(1, int(k_macro))

    all_episode_data = []
    episode_rewards = []
    episode_lengths = []

    for ep in tqdm(range(num_episodes), desc="  - Running Episodes"):
        obs = env.reset()
        done = False
        vC_traj, duty_traj = [], []
        ep_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # (Optional macro-stepping: repeat action k_macro times)
            for _ in range(max(1, int(k_macro))):
                obs, reward, done, infos = env.step(action)
                ep_reward += float(reward[0])
                steps += 1
                info = infos[0]
                vC_traj.append(float(info.get("vC", np.nan)))         # true vC from env info
                duty_traj.append(float(info.get("duty_cmd", np.nan)))  # commanded duty
                if bool(done[0]):
                    break

            done = bool(done[0])

        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        times_arr = np.arange(len(vC_traj), dtype=float) * dt_step
        all_episode_data.append((times_arr, np.array(vC_traj, float), np.array(duty_traj, float)))

    # Aggregate metrics
    metrics = []
    for times_arr, voltages_arr, _duties_arr in all_episode_data:
        m = calculate_performance_metrics(voltages_arr, times_arr, target_voltage, tolerance)
        metrics.append(m)

    def safe_mean(arr): return float(np.nanmean(arr)) if len(arr) else float("nan")
    def safe_std(arr):  return float(np.nanstd(arr))  if len(arr) else float("nan")

    summary = {
        "algo": algo.lower(),
        "episodes": int(num_episodes),
        "training_voltage_noise_std": _extract_training_noise_from_runname(os.path.basename(os.path.dirname(model_path))),
        "evaluation_voltage_noise_std": float(voltage_noise_std),
        "mean_reward": safe_mean(episode_rewards),
        "std_reward":  safe_std(episode_rewards),
        "mean_length": safe_mean(episode_lengths),
        "mean_stabilisation_time_s": safe_mean([m["stabilisation_time"] for m in metrics]),
        "mean_steady_state_error_v": safe_mean([m["steady_state_error"] for m in metrics]),
        "mean_overshoot_v":          safe_mean([m["overshoot"]           for m in metrics]),
        "mean_undershoot_v":         safe_mean([m["undershoot"]          for m in metrics]),
    }

    print("\n  --- Evaluation Summary ---")
    print(f"  Mean Reward             : {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"  Mean Length             : {summary['mean_length']:.1f} steps")
    print(f"  Mean Stabilisation Time : {summary['mean_stabilisation_time_s']*1000:.2f} ms")
    print(f"  Mean Steady-State Error : {summary['mean_steady_state_error_v']*1000:.2f} mV")
    print(f"  Mean Overshoot          : {summary['mean_overshoot_v']*1000:.2f} mV")
    print(f"  Mean Undershoot         : {summary['mean_undershoot_v']*1000:.2f} mV")
    print("  --------------------------")

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_and_save_summary(all_episode_data, target_voltage, tolerance, algo.upper(), out_dir)
    env.close()


def _extract_training_noise_from_runname(run_name: str) -> float:
    try:
        return float(run_name.split("noise_")[-1])
    except Exception:
        return float("nan")


def infer_algo_from_name(folder_name: str) -> str | None:
    for key in ALGO_MAP.keys():
        if folder_name.lower().startswith(key):
            return key
    return None


# ------------------------- CLI orchestration -------------------------

def main():
    p = argparse.ArgumentParser(description="NP-style evaluation for Simulink BBC agents.")
    # Single-model mode
    p.add_argument("--model-path", type=str, help="Path to best_model.zip")
    p.add_argument("--stats-path", type=str, help="Path to <algo>_vec_normalize_final.pkl")
    # Discovery mode
    p.add_argument("--root", default="jax_models_80",
                   help="Root containing trained model folders (algo-prefixed)")

    p.add_argument("--algo", default="all",
                   help=f"Algorithm to evaluate; choices: {list(ALGO_MAP.keys())} or 'all'")
    p.add_argument("--name-contains", default=None,
                   help="Only evaluate folders containing this substring (e.g., 'noise_0.010').")
    p.add_argument("--episodes", type=int, default=5,
                   help="Number of episodes per model.")
    p.add_argument("--tolerance", type=float, default=0.5,
                   help="Voltage tolerance (V) for stabilisation metrics.")
    p.add_argument("--dqn-bins", type=int, default=17,
                   help="Discrete action bins for DQN models.")
    p.add_argument("--eval-noise", type=float, nargs="+", default=[0.0],
                   help="One or more eval-time noise std values to test against.")
    p.add_argument("--model-name", type=str, default="bbcSim",
                   help="Simulink model name (e.g., 'bbcSim').")
    p.add_argument("--macro-k", type=int, default=1,
                   help="Optional macro-step repeat per policy action (for time-axis parity).")

    args = p.parse_args()

    # If explicit single-model is provided, use that path; otherwise discover
    runs = []
    if args.model_path and args.stats_path:
        # Use parent dir name as run_name
        run_name = os.path.basename(os.path.dirname(args.model_path)) or "single_run"
        algo_from_dir = infer_algo_from_name(run_name)
        if args.algo.lower() != "all" and algo_from_dir and algo_from_dir != args.algo.lower():
            print(f"[WARN] Single-run algo '{algo_from_dir}' does not match --algo '{args.algo}'. Proceeding.")

        runs.append({
            "run_name": run_name,
            "model_path": args.model_path,
            "stats_path": args.stats_path,
            "algo": (algo_from_dir or args.algo.lower())
        })
    else:
        print(f"🔍 Searching for models in '{args.root}'...")
        for folder_name in sorted(os.listdir(args.root)):
            run_dir = os.path.join(args.root, folder_name)
            if not os.path.isdir(run_dir):
                continue
            inferred_algo = infer_algo_from_name(folder_name)
            if not inferred_algo:
                continue
            if args.algo.lower() != "all" and inferred_algo != args.algo.lower():
                continue
            if args.name_contains and args.name_contains not in folder_name:
                continue

            model_path = os.path.join(run_dir, "best_model.zip")
            stats_path = os.path.join(run_dir, f"{inferred_algo}_vec_normalize_final.pkl")
            if os.path.exists(model_path) and os.path.exists(stats_path):
                runs.append({
                    "run_name": folder_name,
                    "model_path": model_path,
                    "stats_path": stats_path,
                    "algo": inferred_algo
                })

        if not runs:
            print("No valid models found matching the criteria.")
            return
        print(f"✅ Found {len(runs)} models to evaluate.")

    # Evaluate
    for eval_nl in args.eval_noise:
        print(f"\n{'='*20} Evaluation Pass: Env Noise = {eval_nl:.3f} {'='*20}")
        for run in runs:
            run_name = run["run_name"]
            algo_key = run["algo"].upper()
            out_dir = os.path.join("eval_simulink_runs_80", algo_key,
                                   f"{run['algo'].lower()}_eval_noise_{eval_nl:.3f}",
                                   run_name)
            print(f"\n▶️ Evaluating: {run_name}")
            evaluate_single_model(
                model_path=run["model_path"],
                stats_path=run["stats_path"],
                algo=run["algo"],
                simulink_model=args.model_name,
                out_dir=out_dir,
                num_episodes=args.episodes,
                dqn_bins=args.dqn_bins,
                voltage_noise_std=float(eval_nl),
                tolerance=args.tolerance,
                k_macro=args.macro_k,
            )

if __name__ == "__main__":
    main()
