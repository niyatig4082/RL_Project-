from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_SUMMARY_COLUMNS = [
    "algo",
    "seed",
    "mean_return",
    "success_rate",
    "mean_steps",
    "model_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated result artifacts")
    parser.add_argument("--summary-csv", type=str, default="outputs/multiseed_summary.csv")
    parser.add_argument("--plots-dir", type=str, default="outputs/plots")
    parser.add_argument("--video-path", type=str, default=None)
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty {label}: {path}")


def validate_summary(path: Path) -> None:
    require_file(path, "summary CSV")
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_SUMMARY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Summary CSV missing columns: {missing}")
    if df.empty:
        raise ValueError("Summary CSV has no rows")


def validate_plots(plots_dir: Path) -> None:
    reward_plot = plots_dir / "reward_vs_timesteps.png"
    if reward_plot.exists():
        require_file(reward_plot, "reward plot")
    else:
        print(f"[warn] reward plot not found (allowed when no training curves): {reward_plot}")
    require_file(plots_dir / "final_success_rate.png", "success rate plot")
    require_file(plots_dir / "final_mean_return.png", "mean return plot")


def main() -> None:
    args = parse_args()

    summary_path = Path(args.summary_csv)
    plots_dir = Path(args.plots_dir)

    validate_summary(summary_path)
    validate_plots(plots_dir)

    if args.video_path is not None:
        require_file(Path(args.video_path), "video file")

    print("validation passed")
    print(f"summary_csv: {summary_path}")
    print(f"plots_dir: {plots_dir}")
    if args.video_path is not None:
        print(f"video_path: {args.video_path}")


if __name__ == "__main__":
    main()
