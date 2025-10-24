#!/usr/bin/env python3
"""
Train SB3 agents on the Simulink Buck-Boost env (BBC), with
hyperparameter loading and policy construction mirroring np_bbc_train.py.

Outputs (compatible with simulink_eval_bbc.py discovery):
  jax_models_80/<ALGO>/<algo>_noise_<std>/
    ├─ best_model.zip
    └─ <algo>_vec_normalize_final.pkl

Examples:
  python bbc_train_simulink.py --algo td3 --timesteps 200000 --noise-std 0.000
  python bbc_train_simulink.py --algo dqn --dqn-bins 17 --timesteps 100000 --noise-std 0.010
"""
import torch  
import os
import json
import argparse
import numpy as np
import torch.nn as nn
from typing import Dict, Any

import pandas as pd
from datetime import datetime
from tensorboard.backend.event_processing import event_accumulator

import time
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


from stable_baselines3 import TD3, A2C, SAC, DDPG, PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

from BBCSimulink_env import BBCSimulinkEnv, DiscretizeDutyWrapper

ALGOS = {"td3": TD3, "a2c": A2C, "sac": SAC, "ddpg": DDPG, "ppo": PPO, "dqn": DQN}
OFF_POLICY = {"td3", "sac", "ddpg"} #dqn excluded
ON_POLICY = {"a2c", "ppo"}

ACT_FNS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}

# -------------------- NP-style hyperparameter loading --------------------
# Mirrors the loader in np_bbc_train.py: reads bbc_17_hp_results/<algo>_best_params.json
# If the file has {"best_params": {...}}, unwrap it; else treat flat dict as params.


class FancyTensorboardCallback(BaseCallback):
    def __init__(self, save_steps, save_path_prefix, log_dir, algo_name=None, verbose=0):
        super().__init__(verbose)
        self.save_steps = sorted(save_steps)
        self.save_path_prefix = save_path_prefix
        self.saved_steps = set()
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
        self.timings = {}
        self.start_time = None
        self.pbar = None
        self.log_dir = log_dir            # NEW: for CSV export (parity with IP)
        self.algo_name = algo_name        # NEW: file naming (parity with IP)

    def _on_training_start(self):
        import time
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.total_timesteps = self.model._total_timesteps
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", dynamic_ncols=True)

    def _on_step(self) -> bool:
        import time, os
        self.pbar.update(1)

        # checkpointing (unchanged)
        if (self.num_timesteps in self.save_steps) and (self.num_timesteps not in self.saved_steps):
            self.model.logger.dump(self.num_timesteps)
            base_dir = os.path.dirname(self.save_path_prefix)
            model_file = os.path.join(base_dir, f"best_model_{self.num_timesteps}.zip")
            buffer_file = os.path.join(base_dir, f"replay_buffer_{self.num_timesteps}.pkl")
            self.writer.flush()
            self.model.save(model_file)
            if hasattr(self.model, "save_replay_buffer"):
                self.model.save_replay_buffer(buffer_file)
            now = time.time()
            self.timings[self.num_timesteps] = now - self.last_check_time
            self.last_check_time = now
            print(f"\n Checkpoint at {self.num_timesteps} steps:")
            print(f"    - Model: {model_file}")
            print(f"    - Replay buffer: {buffer_file}")
            print(f"    - Elapsed: {self.timings[self.num_timesteps]:.2f} sec")
            self.saved_steps.add(self.num_timesteps)

        # >>> NEW: log env timings exactly like IP <<<
        for info in self.locals.get("infos", []):
            tdict = info.get("timings")
            if not tdict:
                continue
            for k, v in tdict.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"env_timings/{k}", v, self.num_timesteps)

        # episode logs (unchanged)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.writer.add_scalar("charts/episode_reward", info["episode"]["r"], self.num_timesteps)
                self.writer.add_scalar("charts/episode_length", info["episode"]["l"], self.num_timesteps)
        return True

    # >>> NEW: identical CSV exporter to IP <<<
    def _export_tb_scalars_to_csv(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"tb_scalars_{self.algo_name}.csv")
        ea = event_accumulator.EventAccumulator(self.log_dir)
        ea.Reload()
        rows = []
        for tag in ea.Tags().get("scalars", []):
            for ev in ea.Scalars(tag):
                rows.append({"tag": tag, "step": ev.step, "wall_time": ev.wall_time, "value": ev.value})
        if rows:
            pd.DataFrame(rows).sort_values(["tag", "step"]).to_csv(csv_path, index=False)
            print(f"[TB→CSV] Wrote scalar logs to: {csv_path}")
        else:
            print("[TB→CSV] No scalar tags found in TensorBoard logs.")

    def _on_training_end(self):
        import time
        total_time = time.time() - self.start_time
        self.pbar.close()
        print("\n Training Time Summary:")
        for step, dur in self.timings.items():
            print(f"    {step} steps: {dur:.2f} sec")
        print(f"    Total training time: {total_time:.2f} sec")
        self.writer.flush()
        self.writer.close()
        # >>> NEW: export TB scalars to CSV like IP <<<
        self._export_tb_scalars_to_csv(out_dir="training_timings_bbc")


def load_hyperparameters(algo: str, hp_root: str = "bbc_hp_results") -> Dict[str, Any]:
    path = os.path.join(hp_root, f"{algo.lower()}_best_params.json")
    with open(path, "r") as f:
        blob = json.load(f)
    return blob.get("best_params", blob)


def create_policy_kwargs(params: Dict[str, Any], algo: str) -> Dict[str, Any]:
    net = [int(params["layer_size"])] * int(params["n_layers"])
    act = ACT_FNS[params["activation_fn"].lower()]
    # A2C uses separate pi/vf heads in np_bbc_train.py; others use shared list
    if algo.lower() == "a2c":
        return dict(net_arch=dict(pi=net, vf=net), activation_fn=act)
    return dict(net_arch=net, activation_fn=act)


# ----------------------------- Env factory -----------------------------


def make_env(
    simulink_model: str,
    eval_noise_std: float,
    dt: float,
    frame_skip: int,
    max_episode_time: float,
    grace_steps: int,
    algo: str,
    dqn_bins: int,
):
    def _fn():
        env = BBCSimulinkEnv(
            model_name=simulink_model,
            voltage_noise_std=eval_noise_std,
            dt=dt,
            frame_skip=frame_skip,
            max_episode_time=max_episode_time,
            grace_period_steps=grace_steps,
            use_fast_restart=True,
        )
        if algo.lower() == "dqn":
            env = DiscretizeDutyWrapper(
                env,
                n_bins=dqn_bins,
                low=env.action_space.low[0],
                high=env.action_space.high[0],
            )
        return env

    return _fn


# --------------------------------- Main ---------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(ALGOS.keys()), default="ppo")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--simulink-model", type=str, default="bbcSim")
    p.add_argument("--hp-root", type=str, default="bbc_hp_results")  # NP-style root
    p.add_argument(
        "--root-out", type=str, default="matlab_models_30"
    )  # matches evaluator
    p.add_argument(
        "--noise-std", type=float, default=0.000
    )  # folder naming + sensor noise
    p.add_argument("--dqn-bins", type=int, default=17)

    # Env timing (parity with evaluator and env defaults)
    p.add_argument("--dt", type=float, default=5e-6)
    p.add_argument("--frame-skip", type=int, default=26)
    p.add_argument("--max-episode-time", type=float, default=0.52)
    p.add_argument("--grace-steps", type=int, default=100)
    args = p.parse_args()

    algo = args.algo.lower()
    Algo = ALGOS[algo]

    # Choose device: on-policy -> CPU; others -> GPU if available
    if algo in ON_POLICY:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {algo.upper()} -> {device}")


    # ---------- Output paths that simulink_eval_bbc.py expects ----------
    algo_dir = os.path.join(args.root_out, algo.upper())
    run_dir = os.path.join(algo_dir, f"{algo}_noise_{args.noise_std:0.3f}")
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "best_model.zip")
    stats_path = os.path.join(run_dir, f"{algo}_vec_normalize_final.pkl")

    # ---------- Hyperparams / policy (NP-style) ----------
    hp = load_hyperparameters(algo, args.hp_root)
    policy_kwargs = create_policy_kwargs(hp, algo)

    # ---------- Env + VecNormalize (so eval can load stats) ----------
    env_fn = make_env(
        simulink_model=args.simulink_model,
        eval_noise_std=args.noise_std,
        dt=args.dt,
        frame_skip=args.frame_skip,
        max_episode_time=args.max_episode_time,
        grace_steps=args.grace_steps,
        algo=algo,
        dqn_bins=args.dqn_bins,
    )
    venv = DummyVecEnv([env_fn])
    # norm_obs True; norm_reward True is fine for training; evaluator disables reward norm
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        gamma=float(hp.get("gamma", 0.99)),
        clip_obs=10.0,
    )

    # ---------- Build model from JSON exactly (no PPO batch-size reconstruction here) ----------
    common = dict(
    policy="MlpPolicy",
    env=venv,
    verbose=1,
    learning_rate=float(hp["learning_rate"]),
    gamma=float(hp["gamma"]),
    policy_kwargs=policy_kwargs,
    tensorboard_log=os.path.join("logs_bbc", algo),
    device=device,  # NEW
    )


    # Optional action noise for off-policy with 1D duty action
    action_noise = None
    if algo in {"td3", "ddpg"}:
        sigma = float(hp.get("action_noise_sigma", 0.0))
        if sigma > 0:
            action_noise = NormalActionNoise(mean=np.zeros(1), sigma=sigma * np.ones(1))

    if algo == "td3":
        # Train freq as (k, "step") isn't encoded in NP JSON; use 1 step unless you stored keys
        train_freq = (int(hp.get("train_freq_k", 1)), "step")
        # learning_starts may be absolute or derived; prefer provided value
        learning_starts = int(hp.get("learning_starts", 5000))

        # Support TD3 noise clip multiplier pattern if present
        tpn = float(hp.get("target_policy_noise", 0.2))
        tpn_mult = float(hp.get("target_noise_clip_mult", 2.0))
        tpn_clip = float(hp.get("target_noise_clip", tpn * tpn_mult))

        model = TD3(
            **common,
            buffer_size=int(hp["buffer_size"]),
            batch_size=int(hp["batch_size"]),
            tau=float(hp["tau"]),
            train_freq=train_freq,
            gradient_steps=int(hp.get("gradient_steps", 1)),
            learning_starts=learning_starts,
            policy_delay=int(hp.get("policy_delay", 2)),
            target_policy_noise=tpn,
            target_noise_clip=tpn_clip,
            action_noise=action_noise,
        )
    elif algo == "sac":
        model = SAC(
            **common,
            buffer_size=int(hp["buffer_size"]),
            batch_size=int(hp["batch_size"]),
            tau=float(hp["tau"]),
            ent_coef=hp.get("ent_coef", "auto"),
            train_freq=(int(hp.get("train_freq", 1)), "step"),
            gradient_steps=int(hp.get("gradient_steps", 1)),
            learning_starts=int(hp.get("learning_starts", 1000)),
        )
    elif algo == "ddpg":
        model = DDPG(
            **common,
            buffer_size=int(hp["buffer_size"]),
            batch_size=int(hp["batch_size"]),
            tau=float(hp["tau"]),
            train_freq=(int(hp.get("train_freq", 1)), "step"),
            gradient_steps=int(hp.get("gradient_steps", 1)),
            learning_starts=int(hp.get("learning_starts", 1000)),
            action_noise=action_noise,
        )
    elif algo == "a2c":
        model = A2C(
            **common,
            n_steps=int(hp["n_steps"]),
            ent_coef=float(hp["ent_coef"]),
            vf_coef=float(hp["vf_coef"]),
            max_grad_norm=float(hp["max_grad_norm"]),
            # NP-style extras if present:
            normalize_advantage=bool(hp.get("normalize_advantage", True)),
            use_rms_prop=bool(hp.get("use_rms_prop", True)),
            rms_prop_eps=float(hp.get("rms_prop_eps", 1e-5)),
        )
    elif algo == "ppo":
        model = PPO(
            **common,
            n_steps=int(hp["n_steps"]),
            batch_size=int(hp["batch_size"]),
            n_epochs=int(hp["n_epochs"]),
            clip_range=float(hp["clip_range"]),
            ent_coef=float(hp["ent_coef"]),
            gae_lambda=float(hp["gae_lambda"]),
            vf_coef=float(hp["vf_coef"]),
            max_grad_norm=float(hp["max_grad_norm"]),
        )
    elif algo == "dqn":
        # DQN: use exactly the JSON values
        model = DQN(
            **common,
            buffer_size=int(hp["buffer_size"]),
            batch_size=int(hp["batch_size"]),
            tau=float(hp["tau"]),
            train_freq=int(hp["train_freq"]),  # int, not (k, "step")
            target_update_interval=int(hp["target_update_interval"]),
            exploration_fraction=float(hp["exploration_fraction"]),
            exploration_final_eps=float(hp["exploration_final_eps"]),
            learning_starts=int(hp.get("learning_starts", 5000)),
        )
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    # ---------- Train ----------
    print(f"Training {algo.upper()} for {args.timesteps} timesteps…")

    checkpoint_steps = {
        int(args.timesteps * 0.25),
        int(args.timesteps * 0.5),
        int(args.timesteps * 0.75),
        int(args.timesteps),
    }
    callback = FancyTensorboardCallback(
    save_steps=checkpoint_steps,
    save_path_prefix=model_path,
    log_dir=os.path.join("logs_bbc", algo),
    algo_name=algo,                # NEW (parity with IP)
    )


    print(f"Training {algo.upper()} for {args.timesteps} timesteps…")
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=f"run_noise_{args.noise_std:0.3f}",
        callback=callback,
    )

    # ---------- Save artifacts for evaluator ----------
    model.save(model_path)
    venv.save(stats_path)  # VecNormalize stats for simulink_eval_bbc.py
    print(f"Saved model to {model_path}")
    print(f"Saved VecNormalize stats to {stats_path}")

    # Optional: keep replay buffer for off-policy
    if algo in OFF_POLICY:
        rb_path = os.path.join(run_dir, f"{algo}_replay_buffer.pkl")
        try:
            model.save_replay_buffer(rb_path)
        except Exception:
            pass

    venv.close()
    print("Done.")


if __name__ == "__main__":
    main()
