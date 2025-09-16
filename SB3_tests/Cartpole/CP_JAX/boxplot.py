import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Build dataframe from your extracted results ---
data = [
    # DQN
    ("DQN", 0.0,   0.0,   496.49),
    ("DQN", 0.0,   0.001, 492.73),
    ("DQN", 0.0,   0.01,  496.19),
    ("DQN", 0.0,   0.1,   498.55),
    ("DQN", 0.001, 0.0,   496.41),
    ("DQN", 0.001, 0.001, 492.74),
    ("DQN", 0.001, 0.01,  496.33),
    ("DQN", 0.001, 0.1,   498.56),
    ("DQN", 0.01,  0.0,   496.38),
    ("DQN", 0.01,  0.001, 492.82),
    ("DQN", 0.01,  0.01,  496.25),
    ("DQN", 0.01,  0.1,   498.62),
    ("DQN", 0.1,   0.0,   498.18),
    ("DQN", 0.1,   0.001, 492.95),
    ("DQN", 0.1,   0.01,  495.74),
    ("DQN", 0.1,   0.1,   498.79),
    # TD3
    ("TD3", 0.0,   0.0,   498.44),
    ("TD3", 0.0,   0.001, 498.66),
    ("TD3", 0.0,   0.01,  499.02),
    ("TD3", 0.0,   0.1,   497.39),
    ("TD3", 0.001, 0.0,   498.38),
    ("TD3", 0.001, 0.001, 498.64),
    ("TD3", 0.001, 0.01,  499.01),
    ("TD3", 0.001, 0.1,   497.33),
    ("TD3", 0.01,  0.0,   498.43),
    ("TD3", 0.01,  0.001, 498.68),
    ("TD3", 0.01,  0.01,  499.01),
    ("TD3", 0.01,  0.1,   497.29),
    ("TD3", 0.1,   0.0,   498.63),
    ("TD3", 0.1,   0.001, 497.81),
    ("TD3", 0.1,   0.01,  498.27),
    ("TD3", 0.1,   0.1,   497.35),
]

df = pd.DataFrame(data, columns=["Algorithm", "EnvNoise", "RunNoise", "Reward"])

# --- Plotting ---
fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharey=True)
fig.suptitle("Inverted Pendulum: Reward under Noise Conditions", fontsize=16)

env_levels = sorted(df["EnvNoise"].unique())
run_levels = sorted(df["RunNoise"].unique())

for i, env in enumerate(env_levels):
    for j, run in enumerate(run_levels):
        ax = axes[i, j]
        subset = df[(df["EnvNoise"] == env) & (df["RunNoise"] == run)]
        sns.boxplot(x="Algorithm", y="Reward", data=subset, ax=ax, palette="Set2")
        ax.set_title(f"Env={env}, Run={run}")
        ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel("Reward")
        else:
            ax.set_ylabel("")

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("ip_reward_boxplots.pdf")
plt.show()
