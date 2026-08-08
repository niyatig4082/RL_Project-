from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import miniworld
import numpy as np
import torch
import imageio.v2 as imageio
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

try:
    from train_ddqn_torch import QNet, preprocess_obs
except ModuleNotFoundError:
    from scripts.train_ddqn_torch import QNet, preprocess_obs


def make_sb3_vec_env(frame_stack: int) -> DummyVecEnv:
    env = DummyVecEnv([lambda: gym.make("MiniWorld-FourRooms-v0")])
    env = VecTransposeImage(env)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
    return env


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_sb3_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_sb3(model, episodes: int, frame_stack: int) -> tuple[float, float, float]:
    env = make_sb3_vec_env(frame_stack)
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
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, _ = env.step(action)
            done = bool(done_arr[0])
            truncated = False
            ep_ret += float(reward[0])
            ep_steps += 1

        returns.append(ep_ret)
        steps.append(ep_steps)
        if ep_ret > 0:
            successes += 1

    env.close()
    return float(np.mean(returns)), successes / episodes, float(np.mean(steps))


def write_video(frames: list[np.ndarray], out_path: Path, fps: int = 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)


def infer_ddqn_in_channels(state_dict: dict[str, torch.Tensor]) -> int:
    return int(state_dict["features.0.weight"].shape[1])


def record_sb3_video(model, frame_stack: int, video_dir: Path, video_prefix: str, max_steps: int) -> None:
    env = gym.make("MiniWorld-FourRooms-v0", render_mode="rgb_array")

    obs, _ = env.reset()
    frame_queue: list[np.ndarray] = []
    base = preprocess_obs(obs)
    frames: list[np.ndarray] = [env.render()]
    for _ in range(max(1, frame_stack)):
        frame_queue.append(base)

    done = False
    truncated = False
    steps = 0
    while not (done or truncated) and steps < max_steps:
        stacked = np.concatenate(frame_queue, axis=0)
        action, _ = model.predict(stacked, deterministic=True)
        next_obs, _, done, truncated, _ = env.step(int(action))
        frames.append(env.render())
        frame_queue.pop(0)
        frame_queue.append(preprocess_obs(next_obs))
        steps += 1

    env.close()
    write_video(frames, video_dir / f"{video_prefix}.mp4")


def evaluate_ddqn(path: Path, episodes: int, frame_stack: int) -> tuple[float, float, float]:
    env = gym.make("MiniWorld-FourRooms-v0")
    device = get_torch_device()

    state_dict = torch.load(path, map_location=device)
    in_channels = infer_ddqn_in_channels(state_dict)
    model = QNet(env.action_space.n, in_channels=in_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    returns = []
    successes = 0
    steps = []

    for _ in range(episodes):
        obs, _ = env.reset()
        first = preprocess_obs(obs)
        stack_n = max(1, frame_stack)
        frame_queue = [first for _ in range(stack_n)]
        obs = np.concatenate(frame_queue, axis=0)
        done = False
        truncated = False
        ep_ret = 0.0
        ep_steps = 0

        while not (done or truncated):
            with torch.no_grad():
                action = int(torch.argmax(model(torch.from_numpy(obs).unsqueeze(0).to(device)), dim=1).item())
            next_obs, reward, done, truncated, _ = env.step(action)
            frame_queue.pop(0)
            frame_queue.append(preprocess_obs(next_obs))
            obs = np.concatenate(frame_queue, axis=0)
            ep_ret += reward
            ep_steps += 1

        returns.append(ep_ret)
        steps.append(ep_steps)
        if ep_ret > 0:
            successes += 1

    env.close()
    return float(np.mean(returns)), successes / episodes, float(np.mean(steps))


def record_ddqn_video(path: Path, video_dir: Path, video_prefix: str, max_steps: int, frame_stack: int) -> None:
    env = gym.make("MiniWorld-FourRooms-v0", render_mode="rgb_array")
    device = get_torch_device()
    state_dict = torch.load(path, map_location=device)
    in_channels = infer_ddqn_in_channels(state_dict)
    model = QNet(env.action_space.n, in_channels=in_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    obs, _ = env.reset()
    first = preprocess_obs(obs)
    stack_n = max(1, frame_stack)
    frame_queue = [first for _ in range(stack_n)]
    obs = np.concatenate(frame_queue, axis=0)
    frames: list[np.ndarray] = [env.render()]
    done = False
    truncated = False
    steps = 0

    while not (done or truncated) and steps < max_steps:
        with torch.no_grad():
            action = int(torch.argmax(model(torch.from_numpy(obs).unsqueeze(0).to(device)), dim=1).item())
        next_obs, _, done, truncated, _ = env.step(action)
        frames.append(env.render())
        frame_queue.pop(0)
        frame_queue.append(preprocess_obs(next_obs))
        obs = np.concatenate(frame_queue, axis=0)
        steps += 1

    env.close()
    write_video(frames, video_dir / f"{video_prefix}.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained RL models on MiniWorld FourRooms")
    parser.add_argument("--algo", choices=["ppo", "dqn", "ddqn"], required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="outputs/videos")
    parser.add_argument("--video-prefix", type=str, default=None)
    parser.add_argument("--video-max-steps", type=int, default=500)
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.frame_stack <= 0:
        raise ValueError("--frame-stack must be > 0")
    if args.video_max_steps <= 0:
        raise ValueError("--video-max-steps must be > 0")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    video_dir = Path(args.video_dir)
    video_prefix = args.video_prefix or f"{args.algo}_{model_path.stem}"

    if args.algo == "ppo":
        sb3_device = get_sb3_device()
        print(f"[info] PPO eval using device: {sb3_device}")
        eval_env = make_sb3_vec_env(args.frame_stack)
        model = PPO.load(model_path, env=eval_env, device=sb3_device)
        mean_return, success_rate, mean_steps = evaluate_sb3(model, args.episodes, args.frame_stack)
        if args.record_video:
            record_sb3_video(model, args.frame_stack, video_dir, video_prefix, args.video_max_steps)
        eval_env.close()
    elif args.algo == "dqn":
        sb3_device = get_sb3_device()
        print(f"[info] DQN eval using device: {sb3_device}")
        eval_env = make_sb3_vec_env(args.frame_stack)
        model = DQN.load(model_path, env=eval_env, device=sb3_device)
        mean_return, success_rate, mean_steps = evaluate_sb3(model, args.episodes, args.frame_stack)
        if args.record_video:
            record_sb3_video(model, args.frame_stack, video_dir, video_prefix, args.video_max_steps)
        eval_env.close()
    else:
        print(f"[info] DDQN eval using device: {get_torch_device()}")
        mean_return, success_rate, mean_steps = evaluate_ddqn(model_path, args.episodes, args.frame_stack)
        if args.record_video:
            record_ddqn_video(model_path, video_dir, video_prefix, args.video_max_steps, args.frame_stack)

    print(f"algo: {args.algo}")
    print(f"episodes: {args.episodes}")
    print(f"mean_return: {mean_return:.4f}")
    print(f"success_rate: {success_rate:.4f}")
    print(f"mean_steps: {mean_steps:.2f}")
    if args.record_video:
        video_path = video_dir / f"{video_prefix}.mp4"
        print(f"video_path: {video_path}")


if __name__ == "__main__":
    main()
