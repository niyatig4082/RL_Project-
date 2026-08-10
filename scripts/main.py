from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import torch
from evaluate_model import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO, DQN, and DDQN, evaluate them, and generate summary artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ppo-steps", type=int, default=500_000)
    parser.add_argument("--dqn-steps", type=int, default=250_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--video-episodes", type=int, default=2)
    return parser.parse_args()


def resolve_device(requested: str | None = None) -> str:
    if requested in {None, "auto"}:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested is not None and requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[warn] CUDA was requested but is not available; falling back to CPU")
            return "cpu"
        if ":" in requested:
            device_idx = int(requested.split(":", 1)[1])
            if device_idx >= torch.cuda.device_count():
                print(f"[warn] CUDA device index {device_idx} is not available; falling back to CPU")
                return "cpu"
            torch.cuda.set_device(device_idx)
            return requested
        torch.cuda.set_device(0)
        return "cuda"

    raise ValueError(f"Unsupported device: {requested}")


def validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be >= 0")
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


def require_training_permission(allowed: bool) -> None:
    if not allowed:
        raise PermissionError("Training is disabled by default. Re-run with --allow-training to start training.")


def run_cmd(cmd: list[str], *, label: str | None = None) -> None:
    print("[run]", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip()
        if stripped:
            print(stripped)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_script(script_name: str, args: list[str], *, label: str | None = None) -> None:
    run_cmd([sys.executable, f"scripts/{script_name}", *args], label=label)


def train_and_evaluate(algo: str, seed: int, args: argparse.Namespace, output_dir: Path, log_dir: Path) -> dict[str, float | int | str]:
    if algo == "ppo":
        print(f"[info] Starting PPO training for seed {seed}")
        run_script(
            "train_ppo_sb3.py",
            [
                "--seed",
                str(seed),
                "--total-timesteps",
                str(args.ppo_steps),
                "--eval-freq",
                str(args.eval_freq),
                "--eval-episodes",
                str(args.eval_episodes),
                "--run-name",
                f"seed_{seed}",
                "--output-dir",
                str(output_dir),
                "--log-dir",
                str(log_dir),
                "--device",
                args.device,
                "--allow-training",
            ],
            label="ppo training",
        )
        print(f"[info] PPO training completed for seed {seed}")
        model_path = output_dir / f"ppo_seed_{seed}.zip"
    elif algo == "dqn":
        print(f"[info] Starting DQN training for seed {seed}")
        run_script(
            "train_dqn_sb3.py",
            [
                "--seed",
                str(seed),
                "--total-timesteps",
                str(args.dqn_steps),
                "--eval-freq",
                str(args.eval_freq),
                "--eval-episodes",
                str(args.eval_episodes),
                "--run-name",
                f"seed_{seed}",
                "--output-dir",
                str(output_dir),
                "--log-dir",
                str(log_dir),
                "--device",
                args.device,
                "--allow-training",
            ],
            label="dqn training",
        )
        print(f"[info] DQN training completed for seed {seed}")
        model_path = output_dir / f"dqn_seed_{seed}.zip"
    else:
        print(f"[info] Starting DDQN training for seed {seed}")
        run_script(
            "train_ddqn_sb3.py",
            [
                "--seed",
                str(seed),
                "--total-timesteps",
                str(args.ddqn_steps),
                "--eval-freq",
                str(args.eval_freq),
                "--eval-episodes",
                str(args.eval_episodes),
                "--run-name",
                f"seed_{seed}",
                "--output-dir",
                str(output_dir),
                "--log-dir",
                str(log_dir),
                "--device",
                args.device,
                "--allow-training",
            ],
            label="ddqn training",
        )
        print(f"[info] DDQN training completed for seed {seed}")
        model_path = output_dir / f"ddqn_seed_{seed}.zip"

    return {"algo": algo, "seed": seed, "model_path": str(model_path)}


def validate_outputs(output_dir: Path) -> None:
    required = [
        output_dir / "summary.csv",
        output_dir / "plots" / "reward_vs_timesteps.png",
        output_dir / "plots" / "final_comparison.png",
    ]
    video_dir = output_dir / "videos"
    if not video_dir.exists() or not any(video_dir.glob("*.mp4")):
        required.append(video_dir)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required outputs: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO, DQN, and DDQN, evaluate them, and generate summary artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ppo-steps", type=int, default=100_000)
    parser.add_argument("--dqn-steps", type=int, default=100_000)
    parser.add_argument("--ddqn-steps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--video-episodes", type=int, default=2)
    parser.add_argument("--allow-training", action="store_true", help="Explicitly allow training/evaluation/video generation to run")
    parser.add_argument("--algos", nargs="+", choices=["ppo", "dqn", "ddqn"], default=["ppo", "dqn"])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--algo", choices=["ppo", "dqn", "ddqn"], default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    if args.eval_only:
        if args.algo is None or args.model_path is None or args.episodes is None:
            raise ValueError("--eval-only requires --algo, --model-path, and --episodes")
        output_dir = Path(args.output_dir)
        log_dir = Path(args.log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        device = resolve_device(args.device)
        mean_return, success_rate, mean_steps = evaluate_model(args.algo, Path(args.model_path), args.episodes, output_dir, log_dir, device)
        print(f"algo: {args.algo}")
        print(f"episodes: {args.episodes}")
        print(f"mean_return: {mean_return:.4f}")
        print(f"success_rate: {success_rate:.4f}")
        print(f"mean_steps: {mean_steps:.2f}")
        return

    validate_args(args)
    require_training_permission(args.allow_training)

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[info] main router using device: {device}")

    seed = args.seed
    rows = []
    for algo in args.algos:
        result = train_and_evaluate(algo, seed, args, output_dir, log_dir)
        rows.append(result)

    print("[info] Evaluating trained models")
    summary_path = output_dir / "summary.csv"
    for row in rows:
        print(f"[info] Evaluating {row['algo']}")
        run_script(
            "evaluate_model.py",
            [
                "--algo",
                row["algo"],
                "--model-path",
                str(Path(row["model_path"])),
                "--episodes",
                str(args.eval_episodes),
                "--output-dir",
                str(output_dir),
                "--log-dir",
                str(log_dir),
                "--device",
                device,
                "--seed",
                str(seed),
                "--summary-csv",
                str(summary_path),
            ],
            label="evaluation",
        )

    print(f"Saved summary to {summary_path}")
    for row in rows:
        print(f"[info] Rendering video for {row['algo']}")
        run_script(
            "render_video.py",
            [
                "--algo",
                row["algo"],
                "--model-path",
                str(Path(row["model_path"])),
                "--episodes",
                str(args.video_episodes),
                "--output-dir",
                str(output_dir),
                "--device",
                device,
            ],
            label="video rendering",
        )

    run_script(
        "plot_results.py",
        [
            "--summary-csv",
            str(summary_path),
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(log_dir),
        ],
        label="plot generation",
    )
    validate_outputs(output_dir)
    print("All requested outputs generated.")


if __name__ == "__main__":
    main()
