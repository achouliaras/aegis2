import os
from typing import List, Dict, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

LABELS = {"NoModel": "PPO", 
          "ICM": "ICM", 
          "RND": "RND", 
          "NGU": "NGU", 
          "NovelD":"NovelD", 
          "DEIR":"DEIR",
          "AEGIS":"AEGIS"}

# Custom colors and markers for each algo
COLORS = {"NoModel": "black",
          "ICM": "brown",
          "RND": "orange",
          "NGU": "green",
          "NovelD": "purple",
          "DEIR": "red",
          "AEGIS": "blue"}

MARKERS = {"NoModel": "o",
           "ICM": "s",
           "RND": "D",
           "NGU": "^",
           "NovelD": "v",
           "DEIR": "P",
           "AEGIS": "X"}

# Dict for nicer env titles
ENV_TITLES = {
    "MiniGrid-DoorKey-16x16-v0": "DoorKey-16x16",
    "MiniGrid-DoorKey-8x8-v0": "DoorKey-8x8",
    "MiniGrid-MultiRoom-N4-S5-v0": "MultiRoom-N4-S5",
    "MiniGrid-MultiRoom-N6-v0": "MultiRoom-N6",
    "MiniGrid-FourRooms-v0": "FourRooms",
    "MiniGrid-KeyCorridorS4R3-v0": "KeyCorridor-S4R3",
    "MiniGrid-KeyCorridorS6R3-v0": "KeyCorridor-S6R3",
}

def plot_all_envs_modes(
    envs: List[str],
    modes: List[str],
    algos: List[str],
    base_path: str = "logs",
    out_file_name: str = "all_envs_modes.png",
    n_seeds: int = 10,
    figsize: Tuple[int, int] = (15, 10),
    save_kwargs: Optional[dict] = None,
):
    """
    Create a subplot grid with rows=modes and cols=envs.
    Each subplot shows algo comparison for (env, mode).
    """
    if save_kwargs is None:
        save_kwargs = {}

    n_rows = len(modes)
    n_cols = len(envs)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False, sharey=False)
    if n_rows == 1:  # keep axes iterable
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, mode in enumerate(modes):
        y_pos = 1 - 0.125 - 0.925*i / n_rows  # row center in figure coords
        fig.text(0.01, y_pos, mode, va="center", ha="center",
             rotation=90, fontsize=16, fontweight="bold")
        for j, env in enumerate(envs):
            ax = axes[i][j]

            # ---- copy your existing data loading/averaging logic ----
            averaged: Dict[str, pd.DataFrame] = {}
            required_cols = {"time/total_timesteps", "rollout/ep_info_rew_mean"}

            for algo in algos:
                seed_dfs = []
                for seed in range(n_seeds):
                    csv_path = os.path.join(base_path, env, algo, mode, str(seed), "rollout.csv")
                    if not os.path.exists(csv_path):
                        continue
                    try:
                        df = pd.read_csv(csv_path)
                    except Exception:
                        continue
                    if not required_cols.issubset(df.columns):
                        continue
                    df = df[["time/total_timesteps", "rollout/ep_info_rew_mean"]]
                    seed_dfs.append(df)

                if not seed_dfs:
                    continue

                merged = seed_dfs[0].rename(columns={"rollout/ep_info_rew_mean": f"seed0"})
                for k, df in enumerate(seed_dfs[1:], start=1):
                    merged = pd.merge(
                        merged,
                        df.rename(columns={"rollout/ep_info_rew_mean": f"seed{k}"}),
                        on="time/total_timesteps",
                        how="inner"
                    )

                rewards = merged.drop(columns=["time/total_timesteps"])
                merged["mean"] = rewards.mean(axis=1)
                merged["std"] = rewards.std(axis=1)
                averaged[algo] = merged[["time/total_timesteps", "mean", "std"]]

                # --- plotting ---
                ax.plot(
                    merged["time/total_timesteps"], 
                    merged["mean"],
                    label=LABELS[algo],
                    color=COLORS[algo],
                    marker=MARKERS[algo],
                    markevery=max(len(merged)//10, 1)  # sample markers
                )
                ax.fill_between(
                    merged["time/total_timesteps"],
                    merged["mean"] - merged["std"],
                    merged["mean"] + merged["std"],
                    color=COLORS[algo],
                    alpha=0.2
                )

            # --- formatting ---
            ax.xaxis.set_major_formatter(ScalarFormatter())
            # ax.set_ylim(-0.05, 1.05)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
            if i == n_rows - 1:
                ax.set_xlabel("Steps", fontsize=16)
            if j == 0:
                ax.set_ylabel("Mean Episode Reward", fontsize=16)

            if i == 0:
                ax.set_title(ENV_TITLES.get(env, env), fontsize=18, pad=15)

            # vertical split line logic kept as in your code
            if averaged:
                max_timestep = max(df["time/total_timesteps"].max() for df in averaged.values())
                if "ThreeQuarter" in mode:
                    split_frac = 0.75
                elif "Quarter" in mode:
                    split_frac = 0.25
                elif "Half" in mode:
                    split_frac = 0.50
                else:
                    split_frac = None
                if split_frac is not None:
                    split_step = max_timestep * split_frac
                    ax.axvline(split_step, color="black", linestyle="--")

    # one legend for whole fig
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(algos), fontsize=16, frameon=True)

    plt.tight_layout(rect=[0.02, 0.04, 1, 1])

    os.makedirs("figures", exist_ok=True)
    out_file = f"figures/{out_file_name}"
    plt.savefig(out_file, **save_kwargs)
    plt.close(fig)
    print(f"✅ Saved figure to {out_file}")
    return  # No return of data here, as multiple subplots

# --------------------------
# Example usage (loops outside)
# --------------------------
            # "MiniGrid-DoorKey-8x8-v0",
            # "MiniGrid-KeyCorridorS4R3-v0",
            # "MiniGrid-MultiRoom-N4-S5-v0",

if __name__ == "__main__":
    envs = ["MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-DoorKey-16x16-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-MultiRoom-N4-S5-v0",
            "MiniGrid-MultiRoom-N6-v0",
            "MiniGrid-KeyCorridorS4R3-v0", 
            "MiniGrid-KeyCorridorS6R3-v0",
            ] 
    modes = ["NoPreTrain", "QuarterPreTrain", "HalfPreTrain", "ThreeQuarterPreTrain"] #  
    algos_to_compare = ["NoModel", "ICM", "RND", "NGU", "NovelD", "DEIR", "AEGIS"]

    plot_all_envs_modes(envs, modes, algos_to_compare, base_path="logs", out_file_name="all_envs_modes.png", n_seeds=10,
                        figsize=(28, 12), save_kwargs={"dpi": 400})
