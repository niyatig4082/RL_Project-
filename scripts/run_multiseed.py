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
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--dqn-learning-starts", type=int, default=2_000)
    parser.add_argument("--dqn-buffer-size", type=int, default=200_000)
    parser.add_argument("--dqn-batch-size", type=int, default=128)
    parser.add_argument("--dqn-target-update-interval", type=int, default=2_000)
    parser.add_argument("--dqn-exploration-fraction", type=float, default=0.5)
    parser.add_argument("--dqn-exploration-final-eps", type=float, default=0.02)
    parser.add_argument("--ddqn-learning-starts", type=int, default=2_000)
    parser.add_argument("--ddqn-buffer-size", type=int, default=200_000)
    parser.add_argument("--ddqn-batch-size", type=int, default=128)
    parser.add_argument("--ddqn-target-update-freq", type=int, default=2_000)
    parser.add_argument("--ddqn-epsilon-decay-fraction", type=float, default=0.5)
    parser.add_argument("--ddqn-epsilon-end", type=float, default=0.02)
    parser.add_argument("--ddqn-prioritized-replay", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.seeds:
        raise ValueError("--seeds must include at least one seed")
    if not args.algos:
        raise ValueError("--algos must include at least one algorithm")
    if args.ppo_steps <= 0:
        raise ValueError("--ppo-steps must be > 0")
    if args.dqn_steps <= 0:
        raise ValueError("--dqn-steps must be > 0")
    if args.ddqn_steps <= 0:
        raise ValueError("--ddqn-steps must be > 0")
    if args.eval_freq <= 0:
        raise ValueError("--eval-freq must be > 0")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be > 0")
    if args.frame_stack <= 0:
        raise ValueError("--frame-stack must be > 0")


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_eval_stdout(stdout: str) -> tuple[float, float, float]:
    values = {}
    for line in stdout.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            values[key.strip()] = val.strip()
    required = ["mean_return", "success_rate", "mean_steps"]
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(
            "Could not parse evaluation output. "
            f"Missing fields: {missing}.\nRaw output:\n{stdout}"
        )
    return (
        float(values["mean_return"]),
        float(values["success_rate"]),
        float(values["mean_steps"]),
    )


def evaluate_model(algo: str, model_path: Path, episodes: int, frame_stack: int) -> tuple[float, float, float]:
    cmd = [
        sys.executable,
        "scripts/evaluate_model.py",
        "--algo",
        algo,
        "--model-path",
        str(model_path),
        "--episodes",
        str(episodes),
        "--frame-stack",
        str(frame_stack),
    ]
    print("[eval]", " ".join(cmd))
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if result.stderr:
        print(result.stderr)
    print(result.stdout)
    return parse_eval_stdout(result.stdout)


def main() -> None:
    args = parse_args()
    validate_args(args)

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
                        "--frame-stack",
                        str(args.frame_stack),
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
                        "--frame-stack",
                        str(args.frame_stack),
                        "--learning-starts",
                        str(args.dqn_learning_starts),
                        "--buffer-size",
                        str(args.dqn_buffer_size),
                        "--batch-size",
                        str(args.dqn_batch_size),
                        "--target-update-interval",
                        str(args.dqn_target_update_interval),
                        "--exploration-fraction",
                        str(args.dqn_exploration_fraction),
                        "--exploration-final-eps",
                        str(args.dqn_exploration_final_eps),
                        "--run-name",
                        run_name,
                    ]
                )
                model_path = outputs_dir / f"dqn_{run_name}.zip"
            else:
                ddqn_cmd = [
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
                    "--learning-starts",
                    str(args.ddqn_learning_starts),
                    "--buffer-size",
                    str(args.ddqn_buffer_size),
                    "--batch-size",
                    str(args.ddqn_batch_size),
                    "--target-update-freq",
                    str(args.ddqn_target_update_freq),
                    "--epsilon-decay-fraction",
                    str(args.ddqn_epsilon_decay_fraction),
                    "--epsilon-end",
                    str(args.ddqn_epsilon_end),
                    "--frame-stack",
                    str(args.frame_stack),
                    "--run-name",
                    run_name,
                ]
                if args.ddqn_prioritized_replay:
                    ddqn_cmd.append("--prioritized-replay")
                run_cmd(ddqn_cmd)
                model_path = outputs_dir / f"ddqn_{run_name}.pt"

            mean_return, success_rate, mean_steps = evaluate_model(
                algo,
                model_path,
                args.eval_episodes,
                args.frame_stack,
            )
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
