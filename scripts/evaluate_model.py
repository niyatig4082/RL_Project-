from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

import miniworld


def ensure_miniworld_textures() -> None:
    texture_dir = Path(miniworld.__file__).resolve().parent / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "concrete": (120, 120, 120),
        "concrete_tiles": (140, 140, 140),
        "brick_wall": (170, 90, 60),
        "floor_tiles_bw": (100, 100, 100),
        "asphalt": (60, 60, 60),
    }
    for name, color in required.items():
        target = texture_dir / f"{name}_1.png"
        if target.exists():
            continue
        img = Image.new("RGB", (64, 64), color)
        img.save(target)


def _make_eval_env(render_mode: str | None = None) -> gym.Env:
    ensure_miniworld_textures()
    kwargs = {"max_episode_steps": 250}
    if render_mode:
        kwargs["render_mode"] = render_mode
    return gym.make("MiniWorld-FourRooms-v0", **kwargs)


def make_model_env() -> DummyVecEnv:
    env = DummyVecEnv([lambda: _make_eval_env()])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    return env


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


def evaluate_model(
    algo: str,
    model_path: Path,
    episodes: int,
    output_dir: Path,
    log_dir: Path,
    device: str,
    seed: int,
    summary_csv: Path | None = None,
) -> tuple[float, float, float]:
    resolved_path = model_path
    if not resolved_path.exists() and resolved_path.suffix != ".zip":
        resolved_path = resolved_path.with_suffix(".zip")
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(resolved_path, device=device)
    else:
        from stable_baselines3 import DQN
        model = DQN.load(resolved_path, device=device)

    env = make_model_env()
    returns: list[float] = []
    successes = 0
    steps: list[int] = []

    for episode_idx in range(episodes):
        env.seed(seed + episode_idx)
        obs = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        ep_steps = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            done = bool(done[0])
            truncated = False
            ep_ret += float(reward[0])
            ep_steps += 1
        returns.append(ep_ret)
        steps.append(ep_steps)
        if done and ep_ret > 0:
            successes += 1

    env.close()

    mean_return = float(sum(returns) / len(returns))
    success_rate = successes / episodes
    mean_steps = float(sum(steps) / len(steps))

    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not summary_csv.exists()
        with summary_csv.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["algo", "seed", "model_path", "mean_return", "success_rate", "mean_steps"])
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "algo": algo,
                    "seed": seed,
                    "model_path": str(resolved_path),
                    "mean_return": mean_return,
                    "success_rate": success_rate,
                    "mean_steps": mean_steps,
                }
            )

    print(f"algo: {algo}")
    print(f"episodes: {episodes}")
    print(f"mean_return: {mean_return:.4f}")
    print(f"success_rate: {success_rate:.4f}")
    print(f"mean_steps: {mean_steps:.2f}")
    return mean_return, success_rate, mean_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO or DQN model")
    parser.add_argument("--algo", choices=["ppo", "dqn"], required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / "summary.csv"
    evaluate_model(
        args.algo,
        Path(args.model_path),
        args.episodes,
        output_dir,
        log_dir,
        device,
        args.seed,
        summary_csv,
    )


if __name__ == "__main__":
    main()
