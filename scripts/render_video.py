from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

import gymnasium as gym
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


def render_video(algo: str, model_path: Path, output_dir: Path, episodes: int, device: str) -> None:
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device=device)
    else:
        from stable_baselines3 import DQN
        model = DQN.load(model_path, device=device)

    render_env = _make_eval_env(render_mode="rgb_array")
    model_env = make_model_env()
    frames: list[np.ndarray] = []
    for episode_idx in range(episodes):
        model_env.seed(episode_idx)
        model_obs = model_env.reset()
        render_obs, info = render_env.reset(seed=episode_idx)
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(model_obs, deterministic=True)
            model_obs, reward, done, info = model_env.step(action)
            action_scalar = int(np.asarray(action).reshape(-1)[0])
            render_obs, reward, done, truncated, info = render_env.step(action_scalar)
            frame = render_env.render()
            if frame is None:
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frames.append(np.asarray(frame))
        if frames:
            frames.append(frames[-1])

    output_path = videos_dir / f"{algo}_{model_path.stem}.mp4"
    imageio.mimsave(output_path, frames, fps=20)
    render_env.close()
    model_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render evaluation videos for a trained PPO, DQN, or DDQN model")
    parser.add_argument("--algo", choices=["ppo", "dqn", "ddqn"], required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    render_video(args.algo, Path(args.model_path), output_dir, args.episodes, args.device)


if __name__ == "__main__":
    main()
