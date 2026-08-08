from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import miniworld


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MiniWorld FourRooms observations and actions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--save-dir", type=str, default="outputs/inspect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("MiniWorld-FourRooms-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)

    print(f"obs shape: {obs.shape}")
    print(f"action space: {env.action_space}")

    frame = env.render()
    imageio.imwrite(save_dir / "step_000_reset.png", frame)

    for step in range(1, args.steps + 1):
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        frame = env.render()
        imageio.imwrite(save_dir / f"step_{step:03d}_a{action}.png", frame)
        print(f"step={step:03d} action={action} reward={reward:.3f} done={done} truncated={truncated}")

        if done or truncated:
            obs, _ = env.reset()
            frame = env.render()
            imageio.imwrite(save_dir / f"step_{step:03d}_reset.png", frame)

    env.close()
    print(f"saved frames to: {save_dir}")


if __name__ == "__main__":
    main()
