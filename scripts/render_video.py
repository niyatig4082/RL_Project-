from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack, VecTransposeImage

import gymnasium as gym
import miniworld


def ensure_miniworld_textures() -> None:
    # Use MiniWorld's packaged textures as-is for consistent/default rendering.
    return


def _make_eval_env(render_mode: str | None = None) -> gym.Env:
    ensure_miniworld_textures()
    if render_mode:
        return gym.make("MiniWorld-FourRooms-v0", max_episode_steps=250, render_mode=render_mode)
    return gym.make("MiniWorld-FourRooms-v0", max_episode_steps=250)


def make_model_env() -> VecEnv:
    env = DummyVecEnv([lambda: _make_eval_env()])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    return env


def _rollout_one_episode(model, model_env, render_env, seed: int) -> tuple[list[np.ndarray], bool, int, float]:
    model_env.seed(seed)
    model_obs = model_env.reset()
    render_env.reset(seed=seed)

    done = False
    truncated = False
    episode_return = 0.0
    steps = 0
    episode_frames: list[np.ndarray] = []

    while not (done or truncated):
        action, _ = model.predict(model_obs, deterministic=True)
        model_obs, _, done, _ = model_env.step(action)
        action_scalar = int(np.asarray(action).reshape(-1)[0])
        _, reward, done, truncated, _ = render_env.step(action_scalar)
        episode_return += float(reward)
        steps += 1

        frame = render_env.render()
        if frame is None:
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
        episode_frames.append(np.asarray(frame))

    success = episode_return > 0.0
    return episode_frames, success, steps, episode_return


def render_video(
    algo: str,
    model_path: Path,
    output_dir: Path,
    episodes: int,
    device: str,
    until_success: bool,
    max_attempts: int,
) -> None:
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

    if until_success:
        attempt = 0
        success_count = 0
        while success_count < episodes:
            if max_attempts > 0 and attempt >= max_attempts:
                raise RuntimeError(
                    f"Only found {success_count}/{episodes} successful episodes in {max_attempts} attempts for {algo} model {model_path}."
                )

            episode_frames, success, steps, ep_return = _rollout_one_episode(model, model_env, render_env, seed=attempt)
            print(
                f"[info] attempt={attempt} success={success} steps={steps} episode_return={ep_return:.4f}"
            )
            attempt += 1

            if success:
                success_count += 1
                frames.extend(episode_frames)
                if frames:
                    frames.append(frames[-1])
    else:
        for episode_idx in range(episodes):
            episode_frames, success, steps, ep_return = _rollout_one_episode(
                model, model_env, render_env, seed=episode_idx
            )
            print(
                f"[info] episode={episode_idx} success={success} steps={steps} episode_return={ep_return:.4f}"
            )
            frames.extend(episode_frames)
            if frames:
                frames.append(frames[-1])

    output_path = videos_dir / f"{algo}_{model_path.stem}.mp4"
    # Preserve the original MiniWorld frame size (800x600) instead of auto-resizing to codec macro blocks.
    imageio.mimsave(output_path, frames, fps=20, macro_block_size=1)  # type: ignore[arg-type]
    render_env.close()
    model_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render evaluation videos for a trained PPO, DQN, or DDQN model")
    parser.add_argument("--algo", choices=["ppo", "dqn", "ddqn"], required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--until-success",
        action="store_true",
        help="Keep running episodes until a successful goal-reaching episode is found and save that clip.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum attempts when --until-success is enabled. 0 means unlimited attempts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    render_video(
        args.algo,
        Path(args.model_path),
        output_dir,
        args.episodes,
        args.device,
        args.until_success,
        args.max_attempts,
    )


if __name__ == "__main__":
    main()
