from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_eval_curve(algo: str, run_name: str, log_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    eval_path = log_dir / algo / run_name / "evaluations.npz"
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")
    data = np.load(eval_path)
    timesteps = data["timesteps"]
    results = data["results"]
    return timesteps, results.mean(axis=1)


def generate_plots(summary_path: Path, output_dir: Path, log_dir: Path) -> None:
    df = pd.read_csv(summary_path)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    for algo in sorted(df["algo"].unique()):
        seed_series = df.loc[df["algo"] == algo, "seed"]
        seed_value = int(np.asarray(seed_series).reshape(-1)[0])
        run_name = f"seed_{seed_value}"
        timesteps, rewards = load_eval_curve(algo, run_name, log_dir)
        plt.plot(timesteps, rewards, label=algo)
    plt.xlabel("training timesteps")
    plt.ylabel("evaluation reward")
    plt.title("Reward vs training timesteps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "reward_vs_timesteps.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    summary = df.groupby("algo")["mean_return"].mean()
    summary.plot(kind="bar", color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("mean return")
    plt.title("Final evaluation comparison")
    plt.tight_layout()
    plt.savefig(plots_dir / "final_comparison.png")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reward plots from evaluation summaries")
    parser.add_argument("--summary-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-dir", type=str, default="logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_plots(Path(args.summary_csv), Path(args.output_dir), Path(args.log_dir))


if __name__ == "__main__":
    main()
