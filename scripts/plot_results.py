from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training/evaluation results across seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--algos", nargs="+", choices=["ppo", "dqn", "ddqn"], default=["ppo", "dqn", "ddqn"])
    parser.add_argument("--summary-csv", type=str, default="outputs/multiseed_summary.csv")
    return parser.parse_args()


def load_sb3_curve(algo: str, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    path = Path("logs") / algo / f"seed_{seed}" / "evaluations.npz"
    if not path.exists():
        return None
    data = np.load(path)
    x = data["timesteps"]
    y = data["results"].mean(axis=1)
    return x, y


def load_ddqn_curve(seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    path = Path("outputs") / f"ddqn_seed_{seed}_eval_log.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df["step"].to_numpy(), df["mean_return"].to_numpy()


def aggregate_curves(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(curves) == 0:
        raise ValueError("No curves provided")

    common_x = curves[0][0]
    ys = [curves[0][1]]

    for x, y in curves[1:]:
        interp = np.interp(common_x, x, y)
        ys.append(interp)

    y_stack = np.vstack(ys)
    return common_x, y_stack.mean(axis=0), y_stack.std(axis=0)


def validate_summary_columns(df: pd.DataFrame) -> None:
    required = {"algo", "success_rate", "mean_return"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Summary CSV missing required columns: {missing}")


def main() -> None:
    args = parse_args()

    plots_dir = Path("outputs") / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plotted_any = False

    for algo in args.algos:
        curves: list[tuple[np.ndarray, np.ndarray]] = []

        for seed in args.seeds:
            if algo in {"ppo", "dqn"}:
                curve = load_sb3_curve(algo, seed)
            else:
                curve = load_ddqn_curve(seed)

            if curve is not None:
                curves.append(curve)

        if len(curves) == 0:
            print(f"Skipping {algo}: no curves found")
            continue

        x, mean, std = aggregate_curves(curves)
        label = f"{algo.upper()} (n={len(curves)})"
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        plotted_any = True

    reward_plot_path = plots_dir / "reward_vs_timesteps.png"
    if plotted_any:
        plt.xlabel("Training Timesteps")
        plt.ylabel("Mean Evaluation Return")
        plt.title("FourRooms: Mean Return vs Timesteps")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(reward_plot_path, dpi=180)
        print(f"Saved plot: {reward_plot_path}")
    else:
        print("Skipping reward_vs_timesteps plot: no training curves found")
    plt.close()

    summary_path = Path(args.summary_csv)
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        validate_summary_columns(df)
        if df.empty:
            raise ValueError("Summary CSV is empty")
        agg = (
            df.groupby("algo")
            .agg(
                success_mean=("success_rate", "mean"),
                success_std=("success_rate", "std"),
                return_mean=("mean_return", "mean"),
                return_std=("mean_return", "std"),
            )
            .reset_index()
        )
        agg["success_std"] = agg["success_std"].fillna(0.0)
        agg["return_std"] = agg["return_std"].fillna(0.0)

        plt.figure(figsize=(8, 5))
        plt.bar(agg["algo"].str.upper(), agg["success_mean"], yerr=agg["success_std"], capsize=5)
        plt.ylim(0, 1)
        plt.ylabel("Success Rate")
        plt.title("Final Success Rate by Algorithm")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        success_plot_path = plots_dir / "final_success_rate.png"
        plt.savefig(success_plot_path, dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(agg["algo"].str.upper(), agg["return_mean"], yerr=agg["return_std"], capsize=5)
        plt.ylabel("Mean Return")
        plt.title("Final Return by Algorithm")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return_plot_path = plots_dir / "final_mean_return.png"
        plt.savefig(return_plot_path, dpi=180)
        plt.close()

    if summary_path.exists():
        print(f"Saved plot: {plots_dir / 'final_success_rate.png'}")
        print(f"Saved plot: {plots_dir / 'final_mean_return.png'}")


if __name__ == "__main__":
    main()
