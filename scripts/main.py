from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO and DQN, evaluate them, and generate summary artifacts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--ppo-steps", type=int, default=200_000)
    parser.add_argument("--dqn-steps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
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
    if not args.seeds:
        raise ValueError("--seeds must include at least one seed")
    if args.ppo_steps <= 0:
        raise ValueError("--ppo-steps must be > 0")
    if args.dqn_steps <= 0:
        raise ValueError("--dqn-steps must be > 0")
    if args.eval_freq <= 0:
        raise ValueError("--eval-freq must be > 0")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be > 0")


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_and_evaluate(algo: str, seed: int, args: argparse.Namespace, output_dir: Path, log_dir: Path) -> dict[str, float | int | str]:
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
                f"seed_{seed}",
                "--output-dir",
                str(output_dir),
                "--log-dir",
                str(log_dir),
                "--device",
                args.device,
            ]
        )
        model_path = output_dir / f"ppo_seed_{seed}.zip"
    else:
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
                f"seed_{seed}",
                "--output-dir",
                str(output_dir),
                "--log-dir",
                str(log_dir),
                "--device",
                args.device,
            ]
        )
        model_path = output_dir / f"dqn_seed_{seed}.zip"

    return {"algo": algo, "seed": seed, "model_path": str(model_path)}


def _unwrap_observation(obs: object) -> object:
    if isinstance(obs, tuple):
        return obs[0]
    return obs


def evaluate_model(algo: str, model_path: Path, episodes: int, output_dir: Path, log_dir: Path, device: str) -> tuple[float, float, float]:
    resolved_path = model_path
    if not resolved_path.exists() and resolved_path.suffix != ".zip":
        resolved_path = resolved_path.with_suffix(".zip")
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if algo == "ppo":
        from stable_baselines3 import PPO
        import miniworld
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

        env = DummyVecEnv([lambda: gym.make("MiniWorld-FourRooms-v0")])
        env = VecTransposeImage(env)
        model = PPO.load(resolved_path, env=env, device=device)
        returns = []
        successes = 0
        steps = []
        for _ in range(episodes):
            obs = env.reset()
            done = False
            truncated = False
            ep_ret = 0.0
            ep_steps = 0
            while not (done or truncated):
                obs_value = _unwrap_observation(obs)
                action, _ = model.predict(np.asarray(obs_value), deterministic=True)
                obs, reward, done_arr, _ = env.step(action)
                done = bool(done_arr[0])
                ep_ret += float(reward[0])
                ep_steps += 1
            returns.append(ep_ret)
            steps.append(ep_steps)
            if done and ep_ret > 0:
                successes += 1
        env.close()
        return float(sum(returns) / len(returns)), successes / episodes, float(sum(steps) / len(steps))

    from stable_baselines3 import DQN
    import miniworld
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    env = DummyVecEnv([lambda: gym.make("MiniWorld-FourRooms-v0")])
    env = VecTransposeImage(env)
    model = DQN.load(resolved_path, env=env, device=device)
    returns = []
    successes = 0
    steps = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        ep_steps = 0
        while not (done or truncated):
            obs_value = _unwrap_observation(obs)
            action, _ = model.predict(np.asarray(obs_value), deterministic=True)
            obs, reward, done_arr, _ = env.step(action)
            done = bool(done_arr[0])
            ep_ret += float(reward[0])
            ep_steps += 1
        returns.append(ep_ret)
        steps.append(ep_steps)
        if done and ep_ret > 0:
            successes += 1
    env.close()
    return float(sum(returns) / len(returns)), successes / episodes, float(sum(steps) / len(steps))


def generate_video(algo: str, model_path: Path, output_dir: Path, episodes: int, device: str) -> None:
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device=device)
    else:
        from stable_baselines3 import DQN
        model = DQN.load(model_path, device=device)

    env = gym.make("MiniWorld-FourRooms-v0", render_mode="rgb_array")
    frames = []
    for episode_idx in range(episodes):
        obs, info = env.reset(seed=episode_idx)
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            frame = env.render()
            if frame is None:
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frames.append(np.asarray(frame))
        if frames:
            frames.append(frames[-1])

    output_path = videos_dir / f"{algo}_{model_path.stem}.mp4"
    imageio.mimsave(output_path, frames, fps=20)
    env.close()


def load_eval_curve(algo: str, run_name: str, log_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    eval_path = log_dir / algo / run_name / "evaluations.npz"
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")
    data = np.load(eval_path)
    timesteps = data["timesteps"]
    results = data["results"]
    mean_rewards = results.mean(axis=1)
    return timesteps, mean_rewards


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
    parser = argparse.ArgumentParser(description="Train PPO and DQN, evaluate them, and generate summary artifacts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--ppo-steps", type=int, default=200_000)
    parser.add_argument("--dqn-steps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--video-episodes", type=int, default=2)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default=None)
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

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[info] main router using device: {device}")

    seed = args.seeds[0]
    rows = []
    for algo in ["ppo", "dqn"]:
        result = train_and_evaluate(algo, seed, args, output_dir, log_dir)
        rows.append(result)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "seed", "model_path", "mean_return", "success_rate", "mean_steps"])
        writer.writeheader()
        for row in rows:
            algo = row["algo"]
            model_path = Path(row["model_path"])
            mean_return, success_rate, mean_steps = evaluate_model(algo, model_path, args.eval_episodes, output_dir, log_dir, device)
            writer.writerow({
                "algo": algo,
                "seed": seed,
                "model_path": str(model_path),
                "mean_return": mean_return,
                "success_rate": success_rate,
                "mean_steps": mean_steps,
            })

    print(f"Saved summary to {summary_path}")
    for row in rows:
        generate_video(row["algo"], Path(row["model_path"]), output_dir, args.video_episodes, device)
    generate_plots(summary_path, output_dir, log_dir)
    validate_outputs(output_dir)
    print("All requested outputs generated.")


if __name__ == "__main__":
    main()
