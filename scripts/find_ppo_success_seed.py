from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import miniworld
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage


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
        Image.new("RGB", (64, 64), color).save(target)


def make_env(render_mode: str | None = None):
    ensure_miniworld_textures()
    kwargs: dict[str, object] = {"max_episode_steps": 250}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make("MiniWorld-FourRooms-v0", **kwargs)


def main() -> None:
    model = PPO.load("outputs/ppo_seed_42.zip", device="auto")

    vec_env = DummyVecEnv([lambda: make_env()])
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    env = make_env()

    max_attempts = 5000
    min_steps_for_demo = 25
    found = False

    for seed in range(max_attempts):
        vec_env.seed(seed)
        obs = vec_env.reset()
        env.reset(seed=seed)
        done = False
        truncated = False
        ep_return = 0.0
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = vec_env.step(action)
            action_scalar = int(np.asarray(action).reshape(-1)[0])
            _, reward, done, truncated, _ = env.step(action_scalar)
            ep_return += float(reward)
            steps += 1

        success = ep_return > 0.0
        if success and steps >= min_steps_for_demo:
            print(
                f"SUCCESS seed={seed} steps={steps} episode_return={ep_return:.4f} min_steps={min_steps_for_demo}"
            )
            found = True
            break

    if not found:
        print("NO_SUCCESS_FOUND")

    env.close()
    vec_env.close()


if __name__ == "__main__":
    main()
