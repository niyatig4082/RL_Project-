from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed training and evaluation for RL baselines")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--algos", nargs="+", choices=["ppo", "dqn", "ddqn"], default=["ppo", "dqn", "ddqn"])
    parser.add_argument("--ppo-steps", type=int, default=200_000)
    parser.add_argument("--dqn-steps", type=int, default=200_000)
    parser.add_argument("--ddqn-steps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_eval_stdout(stdout: str) -> tuple[float, float, float]:
    values = {}
    for line in stdout.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            values[key.strip()] = val.strip()
    return (
        float(values["mean_return"]),
        float(values["success_rate"]),
        float(values["mean_steps"]),
    )


def evaluate_model(algo: str, model_path: Path, episodes: int) -> tuple[float, float, float]:
    cmd = [
        sys.executable,
        "scripts/evaluate_model.py",
        "--algo",
        algo,
        "--model-path",
        str(model_path),
        "--episodes",
        str(episodes),
    ]
    print("[eval]", " ".join(cmd))
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print(result.stdout)
    return parse_eval_stdout(result.stdout)


def main() -> None:
    args = parse_args()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = outputs_dir / "multiseed_summary.csv"

    rows: list[dict[str, str | int | float]] = []

    for algo in args.algos:
        for seed in args.seeds:
            run_name = f"seed_{seed}"

            if algo == "ppo":
                run_cmd(
                    [
                        sys.executable,
                        "scripts/train_ppo_sb3.py",
                        "--seed",
                        str(seed),
                        "--total-timesteps",
                        str(args.ppo_steps),
                        "--eval-freq",
                        str(args.eval_freq),
                        "--eval-episodes",
                        str(args.eval_episodes),
                        "--run-name",
                        run_name,
                    ]
                )
                model_path = outputs_dir / f"ppo_{run_name}.zip"
            elif algo == "dqn":
                run_cmd(
                    [
                        sys.executable,
                        "scripts/train_dqn_sb3.py",
                        "--seed",
                        str(seed),
                        "--total-timesteps",
                        str(args.dqn_steps),
                        "--eval-freq",
                        str(args.eval_freq),
                        "--eval-episodes",
                        str(args.eval_episodes),
                        "--run-name",
                        run_name,
                    ]
                )
                model_path = outputs_dir / f"dqn_{run_name}.zip"
            else:
                run_cmd(
                    [
                        sys.executable,
                        "scripts/train_ddqn_torch.py",
                        "--seed",
                        str(seed),
                        "--total-steps",
                        str(args.ddqn_steps),
                        "--eval-every",
                        str(args.eval_freq),
                        "--eval-episodes",
                        str(args.eval_episodes),
                        "--run-name",
                        run_name,
                    ]
                )
                model_path = outputs_dir / f"ddqn_{run_name}.pt"

            mean_return, success_rate, mean_steps = evaluate_model(algo, model_path, args.eval_episodes)
            rows.append(
                {
                    "algo": algo,
                    "seed": seed,
                    "mean_return": mean_return,
                    "success_rate": success_rate,
                    "mean_steps": mean_steps,
                    "model_path": str(model_path),
                }
            )

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["algo", "seed", "mean_return", "success_rate", "mean_steps", "model_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
