from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_eval_data(algo: str, run_name: str, log_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eval_path = log_dir / algo / run_name / "evaluations.npz"
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")
    data = np.load(eval_path)
    timesteps = data["timesteps"]
    results = data["results"]
    lengths = data["ep_lengths"]
    return timesteps, results, lengths


def detect_latest_run_name(algo: str, log_dir: Path) -> str:
    algo_dir = log_dir / algo
    if not algo_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {algo_dir}")

    candidates: list[tuple[float, str]] = []
    for run_dir in algo_dir.iterdir():
        if not run_dir.is_dir():
            continue
        eval_path = run_dir / "evaluations.npz"
        if eval_path.exists():
            candidates.append((eval_path.stat().st_mtime, run_dir.name))

    if not candidates:
        raise FileNotFoundError(f"No evaluation file found under: {algo_dir}")

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    w = max(1, min(window, len(values)))
    series = pd.Series(values)
    return series.rolling(w, min_periods=1).mean().to_numpy()


def build_summary_if_missing(summary_path: Path, log_dir: Path) -> pd.DataFrame:
    if summary_path.exists():
        return pd.read_csv(summary_path)

    rows: list[dict[str, float | str | int]] = []
    for algo in ("ppo", "dqn", "ddqn"):
        try:
            run_name = detect_latest_run_name(algo, log_dir)
            timesteps, results, lengths = load_eval_data(algo, run_name, log_dir)
        except FileNotFoundError:
            continue

        mean_curve = results.mean(axis=1)
        final_idx = len(mean_curve) - 1
        success_rate = float(np.mean(results[final_idx] > 0.0))
        rows.append(
            {
                "algo": algo,
                "seed": 42,
                "model_path": "",
                "mean_return": float(mean_curve[final_idx]),
                "success_rate": success_rate,
                "mean_steps": float(lengths[final_idx].mean()),
            }
        )

    if not rows:
        raise FileNotFoundError("No evaluation checkpoints found to build summary.")

    df = pd.DataFrame(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)
    return df


def generate_plots(summary_path: Path, output_dir: Path, log_dir: Path) -> None:
    df = build_summary_if_missing(summary_path, log_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    run_names: dict[str, str] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for algo in sorted(df["algo"].unique()):
        run_name = detect_latest_run_name(algo, log_dir)
        run_names[algo] = run_name
        curves[algo] = load_eval_data(algo, run_name, log_dir)

    # Plot 1: PPO reference-style scatter + rolling mean
    if "ppo" in curves:
        ppo_timesteps, ppo_results, _ = curves["ppo"]
        ppo_means = ppo_results.mean(axis=1)
        x_scatter = np.repeat(ppo_timesteps, ppo_results.shape[1])
        y_scatter = ppo_results.reshape(-1)

        plt.figure(figsize=(8.6, 4.8))
        plt.scatter(
            x_scatter,
            y_scatter,
            s=8,
            alpha=0.35,
            color="#f4a261",
            edgecolors="none",
            label="episode reward",
        )
        plt.plot(
            ppo_timesteps,
            rolling_mean(ppo_means, window=3),
            color="#1f77b4",
            linewidth=2.2,
            label="rolling mean (evaluation checkpoints)",
        )
        plt.xlabel("Timesteps")
        plt.ylabel("Episode reward")
        plt.title("PPO with frame stacking on MiniWorld-FourRooms")
        plt.grid(alpha=0.25)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / "ppo_training_curve.png", dpi=160)
        plt.close()

    # Plot 2: algorithm comparison curve
    plt.figure(figsize=(8.6, 4.8))
    color_map = {"ppo": "#1f77b4", "dqn": "#d62728", "ddqn": "#ff7f0e"}
    label_map = {"ppo": "PPO", "dqn": "DQN", "ddqn": "DDQN"}
    for algo in ("ppo", "ddqn", "dqn"):
        if algo not in curves:
            continue
        timesteps, results, _ = curves[algo]
        mean_curve = results.mean(axis=1)
        smooth_curve = rolling_mean(mean_curve, window=2)
        plt.plot(
            timesteps,
            smooth_curve,
            label=label_map[algo],
            color=color_map[algo],
            linewidth=2.0,
        )
    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.title("Algorithm comparison on MiniWorld-FourRooms")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "algorithm_comparison.png", dpi=160)
    plt.close()

    # Backward-compatible plot name used by existing workflow.
    plt.figure(figsize=(8.6, 4.8))
    for algo in ("ppo", "ddqn", "dqn"):
        if algo not in curves:
            continue
        timesteps, results, _ = curves[algo]
        mean_curve = results.mean(axis=1)
        plt.plot(timesteps, mean_curve, label=label_map[algo], color=color_map[algo], linewidth=2.0)
    plt.xlabel("training timesteps")
    plt.ylabel("evaluation reward")
    plt.title("Reward vs training timesteps")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "reward_vs_timesteps.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    summary = df.groupby("algo")["mean_return"].mean()
    summary = summary.reindex([algo for algo in ("ppo", "ddqn", "dqn") if algo in summary.index])
    summary.plot(kind="bar", color=["#1f77b4", "#ff7f0e", "#d62728"][: len(summary.index)])
    plt.ylabel("mean return")
    plt.title("Final evaluation comparison")
    plt.tight_layout()
    plt.savefig(plots_dir / "final_comparison.png", dpi=160)
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
