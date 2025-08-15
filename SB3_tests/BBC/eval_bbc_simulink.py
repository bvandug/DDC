import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from BBCSimulink_env import BBCSimulinkEnv

# SB3 (optional)
try:
    from stable_baselines3 import TD3, A2C, SAC, DDPG, PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
    ALGOS = {
        "a2c": A2C,
        "ppo": PPO,
        "ddpg": DDPG,
        "td3": TD3,
        "sac": SAC,
        "dqn": DQN,
    }
except Exception:
    SB3_AVAILABLE = False
    ALGOS = {}
    DummyVecEnv = object  # type: ignore
    VecNormalize = object  # type: ignore


def make_env(model_name: str, dt: float, frame_skip: int, episode_time: float,
             grace_steps: int, target: float, random_target: bool) -> BBCSimulinkEnv:
    return BBCSimulinkEnv(
        model_name=model_name,
        dt=dt,
        frame_skip=frame_skip,
        max_episode_time=episode_time,
        grace_period_steps=grace_steps,
        target_voltage=target,
        random_target=random_target,
        enable_plotting=False,
        use_fast_restart=True,
    )


def load_model_and_env(args):
    """Build (vec)env and optionally load SB3 model + VecNormalize stats."""
    base_env = make_env(
        model_name=args.model_name,
        dt=args.dt,
        frame_skip=args.frame_skip,
        episode_time=args.episode_time,
        grace_steps=args.grace_steps,
        target=args.target,
        random_target=bool(args.random_target),
    )

    model = None
    env_like = base_env
    using_vec = False

    if args.vecnorm_path is not None:
        if not SB3_AVAILABLE:
            raise RuntimeError("--vecnorm-path requires stable-baselines3 installed")
        # Wrap in DummyVecEnv and then load VecNormalize stats
        env_like = DummyVecEnv([lambda: base_env])
        env_like = VecNormalize.load(args.vecnorm_path, env_like)
        env_like.training = False
        env_like.norm_reward = False
        using_vec = True

    if args.algo and args.model_path:
        if not SB3_AVAILABLE:
            raise RuntimeError("--algo/--model-path require stable-baselines3 installed")
        key = args.algo.lower()
        if key not in ALGOS:
            raise ValueError(f"Unsupported algo '{args.algo}'. Use one of: {list(ALGOS.keys())}")
        Model = ALGOS[key]
        # device can be set to 'cpu' to avoid GPU warning for A2C, etc.
        load_kwargs = {"device": args.device} if args.device else {}
        model = Model.load(args.model_path, **load_kwargs)

    return env_like, base_env, model, using_vec


def run_episode_raw(env: BBCSimulinkEnv, model, episode_idx: int, outdir: str,
                    live_plot: bool = False):
    os.makedirs(outdir, exist_ok=True)
    obs, info = env.reset()
    target = float(obs[3])
    T_sw = info.get("T_sw", env.T_sw)

    t_list, vC_list, duty_list, iL_list = [], [], [], []
    total_reward = 0.0

    if live_plot:
        plt.ion()
        fig1 = plt.figure(figsize=(9, 4)); ax1 = fig1.add_subplot(111)
        line_vc, = ax1.plot([], [], label="vC (V)")
        line_vref, = ax1.plot([], [], linestyle=":", label="target (V)")
        ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Voltage (V)"); ax1.legend()
        fig2 = plt.figure(figsize=(9, 3)); ax2 = fig2.add_subplot(111)
        line_duty, = ax2.plot([], [], label="duty")
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Duty (0..1)"); ax2.set_ylim(0, 1)

    t = 0.0
    while True:
        if model is None:
            action = np.asarray([np.random.uniform(0.1, 0.9)], dtype=np.float32)
        else:
            act, _ = model.predict(obs, deterministic=True)
            action = np.asarray(act, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        vC = float(info.get("vC", obs[0]))
        duty = float(info.get("duty_cmd", action[0]))
        iL = info.get("iL", None)

        t_list.append(t)
        vC_list.append(vC)
        duty_list.append(duty)
        iL_list.append(np.nan if iL is None else float(iL))

        if live_plot:
            line_vc.set_data(t_list, vC_list)
            line_vref.set_data(t_list, [target] * len(t_list))
            ax1.relim(); ax1.autoscale_view()
            line_duty.set_data(t_list, duty_list)
            ax2.relim(); ax2.autoscale_view(); ax2.set_ylim(0, 1)
            plt.pause(0.001)

        t += T_sw
        if terminated or truncated:
            break

    return _finalize_plots(outdir, episode_idx, t_list, vC_list, duty_list, iL_list, target, total_reward)


def run_episode_vec(venv, model, base_env: BBCSimulinkEnv, episode_idx: int, outdir: str,
                    live_plot: bool = False):
    os.makedirs(outdir, exist_ok=True)

    # Reset vector env (n_envs=1)
    obs = venv.reset()
    # Gymnasium VecEnv returns (obs, infos) in some wrappers; support both
    if isinstance(obs, tuple):
        obs, infos = obs
        info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) else infos
    else:
        # Pull info by resetting underlying env for target/T_sw
        _, info0 = base_env.reset()

    # Extract timing/target from base env
    target = float(base_env.target_voltage)
    T_sw = getattr(base_env, "T_sw", 1.0)

    t_list, vC_list, duty_list, iL_list = [], [], [], []
    total_reward = 0.0

    if live_plot:
        plt.ion()
        fig1 = plt.figure(figsize=(9, 4)); ax1 = fig1.add_subplot(111)
        line_vc, = ax1.plot([], [], label="vC (V)")
        line_vref, = ax1.plot([], [], linestyle=":", label="target (V)")
        ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Voltage (V)"); ax1.legend()
        fig2 = plt.figure(figsize=(9, 3)); ax2 = fig2.add_subplot(111)
        line_duty, = ax2.plot([], [], label="duty")
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Duty (0..1)"); ax2.set_ylim(0, 1)

    t = 0.0
    while True:
        if model is None:
            action = np.asarray([[np.random.uniform(0.1, 0.9)]], dtype=np.float32)
        else:
            act, _ = model.predict(obs, deterministic=True)
            # ensure shape (n_envs, action_dim)
            action = np.asarray(act, dtype=np.float32)
            if action.ndim == 1:
                action = action.reshape(1, -1)

        step_out = venv.step(action)
        # Support both Gym and Gymnasium VecEnv return signatures
        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            terminated = bool(dones[0])
            truncated = False
            info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) else {}
            reward_scalar = float(rewards[0])
        else:
            obs, rewards, terminateds, truncateds, infos = step_out
            terminated = bool(terminateds[0]); truncated = bool(truncateds[0])
            info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) else {}
            reward_scalar = float(rewards[0])

        total_reward += reward_scalar

        # Pull raw telemetry from base_env (since venv normalizes obs, not info)
        vC = float(base_env.prev_vC)
        duty = float(base_env.prev_duty)
        iL = base_env.last_iL

        t_list.append(t)
        vC_list.append(vC)
        duty_list.append(duty)
        iL_list.append(np.nan if iL is None else float(iL))

        if live_plot:
            line_vc.set_data(t_list, vC_list)
            line_vref.set_data(t_list, [target] * len(t_list))
            ax1.relim(); ax1.autoscale_view()
            line_duty.set_data(t_list, duty_list)
            ax2.relim(); ax2.autoscale_view(); ax2.set_ylim(0, 1)
            plt.pause(0.001)

        t += T_sw
        if terminated or truncated:
            break

    return _finalize_plots(outdir, episode_idx, t_list, vC_list, duty_list, iL_list, target, total_reward)


def _finalize_plots(outdir, episode_idx, t_list, vC_list, duty_list, iL_list, target, total_reward):
    t_arr = np.asarray(t_list); vC_arr = np.asarray(vC_list)
    duty_arr = np.asarray(duty_list); iL_arr = np.asarray(iL_list)

    fig_v = plt.figure(figsize=(10, 4)); axv = fig_v.add_subplot(111)
    axv.plot(t_arr, vC_arr, label="vC (V)")
    axv.plot(t_arr, np.full_like(t_arr, target), linestyle=":", label="target (V)")
    axv.set_title(f"BBC Simulink — Episode {episode_idx} (reward={total_reward:.2f})")
    axv.set_xlabel("Time (s)"); axv.set_ylabel("Voltage (V)"); axv.legend(loc="best")
    fig_v.tight_layout(); v_path = os.path.join(outdir, f"ep{episode_idx:02d}_voltage.png"); fig_v.savefig(v_path, dpi=160)

    fig_d = plt.figure(figsize=(10, 3.2)); axd = fig_d.add_subplot(111)
    axd.plot(t_arr, duty_arr, label="duty")
    axd.set_xlabel("Time (s)"); axd.set_ylabel("Duty (0..1)"); axd.set_ylim(0.0, 1.0); axd.legend(loc="best")
    fig_d.tight_layout(); d_path = os.path.join(outdir, f"ep{episode_idx:02d}_duty.png"); fig_d.savefig(d_path, dpi=160)

    i_path = None
    if not np.all(np.isnan(iL_arr)):
        fig_i = plt.figure(figsize=(10, 3.2)); axi = fig_i.add_subplot(111)
        axi.plot(t_arr, iL_arr, label="iL (A)"); axi.set_xlabel("Time (s)"); axi.set_ylabel("Inductor Current (A)"); axi.legend(loc="best")
        fig_i.tight_layout(); i_path = os.path.join(outdir, f"ep{episode_idx:02d}_iL.png"); fig_i.savefig(i_path, dpi=160)

    npz_path = os.path.join(outdir, f"ep{episode_idx:02d}.npz")
    np.savez_compressed(npz_path, t=t_arr, vC=vC_arr, duty=duty_arr, iL=iL_arr,
                        target=np.array([target], dtype=float), total_reward=np.array([total_reward], dtype=float))

    return {
        "plots": {"voltage": v_path, "duty": d_path, "iL": i_path},
        "npz": npz_path,
        "reward": total_reward,
        "steps": len(t_arr),
    }


def main():
    parser = argparse.ArgumentParser(description="Run BBC Simulink env for N episodes and plot (VecNormalize-aware)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="plots_bbc_simulink")
    parser.add_argument("--live-plot", action="store_true")

    # Env args
    parser.add_argument("--model-name", type=str, default="bbcSim")
    parser.add_argument("--dt", type=float, default=5e-6)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--episode-time", type=float, default=0.2)
    parser.add_argument("--grace-steps", type=int, default=100)
    parser.add_argument("--target", type=float, default=-30.0)
    parser.add_argument("--random-target", action="store_true")

    # Policy / model args
    parser.add_argument("--algo", type=str, default=None, help="a2c, ppo, ddpg, td3, sac, dqn")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="force SB3 model device, e.g. 'cpu'")

    # VecNormalize stats
    parser.add_argument("--vecnorm-path", type=str, default=None, help="Path to saved VecNormalize stats (.pkl)")

    args = parser.parse_args()

    env_like, base_env, model, using_vec = load_model_and_env(args)

    # Run episodes
    summary = []
    for ep in range(1, args.episodes + 1):
        if using_vec:
            res = run_episode_vec(env_like, model, base_env, ep, args.outdir, live_plot=args.live_plot)
        else:
            res = run_episode_raw(base_env, model, ep, args.outdir, live_plot=args.live_plot)
        summary.append((ep, res["reward"], res["steps"]))
        print(f"Episode {ep}: reward={res['reward']:.2f}, steps={res['steps']}")

    # Summary overlay of vC
    fig_sum = plt.figure(figsize=(11, 5)); ax_sum = fig_sum.add_subplot(111)
    for ep in range(1, args.episodes + 1):
        data = np.load(os.path.join(args.outdir, f"ep{ep:02d}.npz"))
        ax_sum.plot(data["t"], data["vC"], label=f"ep {ep}")
    last = np.load(os.path.join(args.outdir, f"ep{args.episodes:02d}.npz"))
    target = float(last["target"][0])
    ax_sum.plot(last["t"], np.full_like(last["t"], target), linestyle=":", label="target")
    ax_sum.set_title("BBC Simulink — vC across episodes")
    ax_sum.set_xlabel("Time (s)"); ax_sum.set_ylabel("Voltage (V)"); ax_sum.legend(loc="best", ncol=2)
    fig_sum.tight_layout(); fig_sum.savefig(os.path.join(args.outdir, "summary_voltage.png"), dpi=160)

    print("Summary:")
    for ep, rew, steps in summary:
        print(f"  ep {ep:02d}: reward={rew:.2f}, steps={steps}")

    # Ensure MATLAB engine closes
    base_env.close()


if __name__ == "__main__":
    main()
