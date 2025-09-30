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
          "AEGIS_5e-2":r"$\alpha = 0.05$",
          "AEGIS_1e-3":r"$\alpha = 0.001$",
          "AEGIS_5e-1":r"$\alpha = 0.5$",
          "AEGIS_1e-1":r"$\alpha = 0.1$",
          "AEGIS_5e-3":r"$\alpha = 0.005$",
          "AEGIS_1":r"$\alpha = 1$",
          "AEGIS_global_only": "Only global int reward",
          "AEGIS_local_only": "Only local int reward",
          "AEGIS_alt": "AEGIS Alternating updates",
          "AEGIS_plus": "AEGIS (local + global)",
          "AEGIS": "AEGIS (local * global)",
          "AEGIS_forward": "AEGIS (forward model)",
          "AEGIS_forward_diff": "AEGIS (forward novelty diff)",
          }
        #   "AEGIS":r"AEGIS ($\alpha = 0.01$)",

def plot_algorithms_for_env(
    env: str,
    mode: str,
    algos: List[str],
    out_filename: str,
    base_path: str = "logs",
    n_seeds: int = 10,
    figsize: Tuple[int,int] = (10, 6),
    save_kwargs: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    For a given env and mode, load logs/{env}/{algo}/{mode}/{seed}/rollout.csv
    for each algo and seed, average across seeds, plot mean ± std,
    save the figure to out_filename, and return a dict of averaged DataFrames.

    Parameters
    ----------
    env : str
        Environment name (kept outside in caller loops).
    mode : str
        Mode name (kept outside in caller loops).
    algos : list[str]
        List of algorithm folder names to compare (this loop is inside the function).
    out_filename : str
        Path to save the resulting figure (extension determines format, e.g. .png, .pdf).
    base_path : str
        Base logs folder (default "logs").
    n_seeds : int
        Number of seeds (folders 0 ... n_seeds-1).
    figsize : tuple
        Figure size in inches.
    save_kwargs : dict, optional
        Extra kwargs to pass to plt.savefig (e.g. dpi=300).
    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from algorithm name to a DataFrame with columns
        ['time/total_timesteps', 'mean', 'std'].
    """
    if save_kwargs is None:
        save_kwargs = {}

    fig, ax = plt.subplots(figsize=figsize)

    averaged: Dict[str, pd.DataFrame] = {}
    required_cols = {"time/total_timesteps", "rollout/ep_rew_mean"}

    for algo in algos:
        seed_dfs = []
        for seed in range(n_seeds):
            csv_path = os.path.join(base_path, env, algo, mode, str(seed), "rollout.csv")
            if not os.path.exists(csv_path):
                print(f"⚠️ Missing file: {csv_path}")
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"❌ Error reading {csv_path}: {e}")
                continue
            if not required_cols.issubset(df.columns):
                print(f"⚠️ Missing columns in {csv_path}")
                continue
            # Keep only needed columns
            df = df[["time/total_timesteps", "rollout/ep_info_rew_mean"]]
            seed_dfs.append(df)

        if not seed_dfs:
            print(f"ℹ️ No valid data for algo={algo}")
            continue

        # Merge on timesteps (inner join across seeds)
        merged = seed_dfs[0].rename(columns={"rollout/ep_info_rew_mean": f"seed0"})
        for i, df in enumerate(seed_dfs[1:], start=1):
            merged = pd.merge(
                merged,
                df.rename(columns={"rollout/ep_info_rew_mean": f"seed{i}"}),
                on="time/total_timesteps",
                how="inner"
            )

        # Compute mean/std across seeds
        rewards = merged.drop(columns=["time/total_timesteps"])
        merged["mean"] = rewards.mean(axis=1)
        merged["std"] = rewards.std(axis=1)

        averaged[algo] = merged[["time/total_timesteps", "mean", "std"]]

        # Plot with std band
        ax.plot(merged["time/total_timesteps"], merged["mean"], label=LABELS[algo])
        ax.fill_between(
            merged["time/total_timesteps"],
            merged["mean"] - merged["std"],
            merged["mean"] + merged["std"],
            alpha=0.2
        )

    if not averaged:
        print(f"ℹ️ No valid data found for env='{env}', mode='{mode}'. Nothing plotted.")
        plt.close(fig)
        return averaged

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(16)   # change scientific notation text size
    ax.tick_params(axis="both", which="major", labelsize=16)  # bigger font for major ticks
    ax.tick_params(axis="both", which="minor", labelsize=14)  # optional, minor ticks
    ax.set_xlabel("Total Timesteps", fontsize=16)
    ax.set_ylabel("Episode Reward Mean", fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{env} — {mode} — feature ablation", fontsize=18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1), borderaxespad=0.0, fontsize=16)
    ax.grid(True)
    plt.tight_layout()

    # --- Add vertical line for pretraining/training split ---
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
        ax.axvline(split_step, color="black", linestyle="--", label="Pretraining")

        # Annotate text
        ymin, ymax = ax.get_ylim()
        ax.text(split_step * 0.5, ymax * 0.975, "Pretraining",
                ha="center", va="top", color="black", fontsize=16, fontweight="bold")
        ax.text(split_step + (max_timestep - split_step) * 0.5, ymax * 0.975, "Training",
                ha="center", va="top", color="black", fontsize=16, fontweight="bold")
        
    try:
        # os.makedirs(os.path.dirname(f"figures/{mode}/"+out_filename), exist_ok=True)
        # plt.savefig(f"figures/{mode}/"+out_filename, **save_kwargs)
        # print(f"✅ Saved figure to 'figures/{mode}/"+out_filename)
        os.makedirs(os.path.dirname(f"figures/"+out_filename), exist_ok=True)
        plt.savefig(f"figures/"+out_filename, **save_kwargs)
        print(f"✅ Saved figure to 'figures/"+out_filename)
    except Exception as e:
        print(f"❌ Error saving figure '{out_filename}': {e}")
    finally:
        plt.close(fig)

    return averaged

# --------------------------
# Example usage (loops outside)
# --------------------------
            # "MiniGrid-DoorKey-8x8-v0", 
            # "MiniGrid-DoorKey-16x16-v0", 
            # "MiniGrid-FourRooms-v0",
            # "MiniGrid-MultiRoom-N4-S5-v0",
            # "MiniGrid-MultiRoom-N6-v0", 
            # "MiniGrid-KeyCorridorS4R3-v0",
            # "MiniGrid-KeyCorridorS6R3-v0",
if __name__ == "__main__":
    envs = ["MiniGrid-DoorKey-8x8-v0",
            ] 
    modes = ["HalfPreTrain"] # "NoPreTrain", "QuarterPreTrain", "HalfPreTrain", "ThreeQuarterPreTrain"
    algos_to_compare = ["AEGIS", "AEGIS_plus","AEGIS_global_only", "AEGIS_local_only", "AEGIS_alt", "AEGIS_forward", "AEGIS_forward_diff"]

    for mode in modes:
        for env in envs:
            out_file = f"{env.replace('-', '_')}_{mode}_ablation.png"
            data = plot_algorithms_for_env(env=env,
                                           mode=mode,
                                           algos=algos_to_compare,
                                           out_filename=out_file,
                                           base_path="logs",
                                           n_seeds=10,
                                           figsize=(12, 6),
                                           save_kwargs={"dpi": 200})
            # `data` is a dict {algo: DataFrame} you can use for further analysis if desired.
