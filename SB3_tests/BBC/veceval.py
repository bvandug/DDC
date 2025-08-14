# eval_bbc_a2c.py — Standalone evaluator (with action debug)
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize as VN
from stable_baselines3.common.monitor import Monitor

from np_bbc_env import JAXBuckBoostConverterEnv


def make_env(frame_skip: int, target_voltage: float):
    def _thunk():
        env = JAXBuckBoostConverterEnv(
            max_episode_steps=4000,
            frame_skip=20,          # MUST match training
            grace_period_steps=50,
            dt=5e-6,
            target_voltage=target_voltage,
        )
        return Monitor(env)
    return _thunk


def infer_vecnorm_path(model_path: str) -> str | None:
    d = os.path.dirname(model_path)
    # prefer filenames that look like vec-normalize stats
    patterns = ["*vec_normalize*.pkl", "*vecnormalize*.pkl", "*.pkl"]
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(d, pat))):
            return p
    return None


def ideal_hold_duty(vin: float, vref: float) -> float:
    # CCM: D* = |Vref| / (|Vref| + Vin)
    return abs(vref) / (abs(vref) + vin)


def eval_and_plot(model_path: str,
                  vecnorm_path: str | None,
                  deterministic: bool,
                  max_steps: int,
                  save_dir: str,
                  raw_reward: bool,
                  frame_skip: int,
                  target_voltage: float) -> None:

    model = A2C.load(model_path, device="cpu")

    # Build env
    eval_env = DummyVecEnv([make_env(frame_skip, target_voltage)])

    # Load VecNormalize stats (REQUIRED if you trained with norm_obs=True)
    used_vecnorm = False
    if vecnorm_path is None:
        vecnorm_path = infer_vecnorm_path(model_path)

    if vecnorm_path and os.path.isfile(vecnorm_path):
        eval_env = VN.load(vecnorm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = not raw_reward
        used_vecnorm = True
        print(f"[OK] Loaded VecNormalize stats: {vecnorm_path}")
    else:
        print("[WARN] VecNormalize stats NOT found. If training used normalized "
              "observations, results may be meaningless. Pass --vecnorm PATH.")

    # For diagnostics
    low = float(eval_env.action_space.low[0])
    high = float(eval_env.action_space.high[0])

    obs = eval_env.reset()
    rewards, duties, vCs, iLs = [], [], [], []
    mu_list, sd_list, diffs = [], [], []
    bound_hits = 0
    mu_bound_hits = 0
    ep_return = 0.0

    steps = 0
    bb = None  # will hold inner env for plotting lines
    while steps < max_steps:
        # action used for env step
        action, _ = model.predict(obs, deterministic=deterministic)
        act = float(action[0][0])

       # --- CORRECT: get deterministic mean using latent -> dist -> mode -> scale ---
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
            features = model.policy.extract_features(obs_t)
            latent_pi, _ = (model.policy.mlp_extractor(features)
                            if hasattr(model.policy, "mlp_extractor") else (features, None))
            dist = model.policy._get_action_dist_from_latent(latent_pi)  # SquashedDiagGaussian
            mode = dist.mode()                                           # in [-1, 1], already tanh-squashed
            mu_rescaled_t = model.policy.scale_action(mode)              # to [low, high]
            mu_rescaled = float(mu_rescaled_t.cpu().numpy()[0, 0])

            # optional: raw std (pre-squash) if you want to log it
            sd_raw = (float(dist.distribution.stddev.detach().cpu().numpy()[0, 0])
                    if hasattr(dist, "distribution") else float("nan"))
        # -------------------------------------------------------------------------


        # store debug stats
        mu_list.append(float(mu_rescaled))
        sd_list.append(float(sd_raw))
        diff = abs(mu_rescaled - act)
        diffs.append(diff)
        EPS = 5e-4
        if act <= low + EPS or act >= high - EPS:
            bound_hits += 1
        if mu_rescaled <= low + EPS or mu_rescaled >= high - EPS:
            mu_bound_hits += 1


        # step env
        obs, r, dones, infos = eval_env.step(action)  # dones is a vector
        r_scalar = float(r.mean())
        ep_return += r_scalar

        rewards.append(r_scalar)
        duties.append(act)

        # Peek inner env (DummyVecEnv + Monitor)
        inner_monitor = eval_env.venv.envs[0]
        bb = inner_monitor.env
        vCs.append(float(bb.state[1]))
        iLs.append(float(bb.state[0]))

        steps += 1
        # end when the single env finishes
        if bool(dones[0]) or any(info.get("episode") is not None for info in infos):
            break

    # safety: if episode ended before we set bb for some reason
    if bb is None:
        bb = eval_env.venv.envs[0].env

    ep_len = len(rewards)
    print(f"Episode length: {ep_len}")
    print(f"Episode return ({'RAW' if raw_reward else 'NORMALIZED'}): {ep_return:.2f}")

    # Helpful summaries
    vin = float(getattr(bb, "Vin", 48.0))
    d_star = ideal_hold_duty(vin, bb.target_voltage)
    avg_mu_last = float(np.mean(mu_list[-500:])) if ep_len > 0 else float("nan")
    avg_act_last = float(np.mean(duties[-500:])) if ep_len > 0 else float("nan")
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    max_diff = float(np.max(diffs)) if diffs else 0.0
    frac_act_bound = bound_hits / max(1, ep_len)
    frac_mu_bound = mu_bound_hits / max(1, ep_len)

    print(f"Ideal holding duty D*: {d_star:.4f}  (Vin={vin:.2f} V, Vref={bb.target_voltage:.2f} V)")
    print(f"Mean(policy, squashed) over last 500: {avg_mu_last:.4f}")
    print(f"Mean(action)          over last 500: {avg_act_last:.4f}")
    print(f"Mean |action - mean|: {avg_diff:.3e}, max: {max_diff:.3e}")
    print(f"Action on bounds: {frac_act_bound:.2%}   |   Mean on bounds: {frac_mu_bound:.2%}")

    # === Save CSV ===
    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, "eval_trace.csv")
    with open(out_csv, "w") as f:
        f.write("step,reward,duty,vC,iL,mu,sd,diff\n")
        for i in range(ep_len):
            f.write(f"{i},{rewards[i]},{duties[i]},{vCs[i]},{iLs[i]},{mu_list[i]},{sd_list[i]},{diffs[i]}\n")
    print(f"Saved per-step CSV to: {out_csv}")

    # === Plots ===
    plt.figure(); plt.title(f"Per-step {'raw' if raw_reward else 'normalized'} reward")
    plt.plot(rewards); plt.xlabel("Step"); plt.ylabel("Reward")
    plt.savefig(os.path.join(save_dir, "eval_rewards.png"))

    plt.figure(); plt.title("Output voltage vC vs target and |bounds|")
    plt.plot(vCs)
    plt.axhline(y=bb.target_voltage, linestyle="--")
    plt.axhline(y= bb.V_OUT_MAX, linestyle=":")
    plt.axhline(y=-bb.V_OUT_MAX, linestyle=":")
    plt.axhline(y= bb.V_OUT_MIN, linestyle=":")
    plt.axhline(y=-bb.V_OUT_MIN, linestyle=":")
    plt.xlabel("Step"); plt.ylabel("vC [V]")
    plt.savefig(os.path.join(save_dir, "eval_vc.png"))

    plt.figure(); plt.title("Inductor current iL")
    plt.plot(iLs); plt.xlabel("Step"); plt.ylabel("iL [A]")
    plt.savefig(os.path.join(save_dir, "eval_iL.png"))

    plt.figure(); plt.title("Duty cycle")
    plt.plot(duties); plt.xlabel("Step"); plt.ylabel("Duty")
    plt.savefig(os.path.join(save_dir, "eval_duty.png"))

    plt.figure(); plt.title("Policy mean (squashed) and action (last 500)")
    n = min(500, len(mu_list))
    plt.plot(mu_list[-n:], label="policy mean (squashed+rescaled)")
    plt.plot(duties[-n:], label="action (after sampling/clipping)")
    plt.legend(); plt.xlabel("Step (last 500)"); plt.ylabel("Duty")
    plt.savefig(os.path.join(save_dir, "eval_mu_vs_action.png"))

    plt.figure(); plt.title("|action - mean| (last 500)")
    n = min(500, len(diffs))
    plt.plot(diffs[-n:])
    plt.xlabel("Step (last 500)"); plt.ylabel("|diff|")
    plt.savefig(os.path.join(save_dir, "eval_action_minus_mean.png"))

    plt.show()
    print(f"Saved plots to: {save_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to A2C model .zip")
    p.add_argument("--vecnorm", default=None, help="Path to VecNormalize .pkl")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--save-dir", default="./eval_plots")
    # Default to RAW rewards; pass --normalized to view normalized rewards
    p.add_argument("--normalized", action="store_true", help="Report NORMALIZED rewards instead of raw")
    p.add_argument("--frame-skip", type=int, default=20, help="Must match training")
    p.add_argument("--target-voltage", type=float, default=-30.0)
    args = p.parse_args()

    raw_reward = not args.normalized  # default RAW=True unless you pass --normalized

    eval_and_plot(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        raw_reward=raw_reward,
        frame_skip=args.frame_skip,
        target_voltage=args.target_voltage,
    )


if __name__ == "__main__":
    main()
