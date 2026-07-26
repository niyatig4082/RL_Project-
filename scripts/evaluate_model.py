from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN, PPO

from train_ddqn_torch import QNet, preprocess_obs


def evaluate_sb3(model, episodes: int) -> tuple[float, float, float]:
    env = gym.make("MiniWorld-FourRooms-v0")
    returns = []
    successes = 0
    steps = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        ep_steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_ret += reward
            ep_steps += 1

        returns.append(ep_ret)
        steps.append(ep_steps)
        if ep_ret > 0:
            successes += 1

    env.close()
    return float(np.mean(returns)), successes / episodes, float(np.mean(steps))


def evaluate_ddqn(path: Path, episodes: int) -> tuple[float, float, float]:
    env = gym.make("MiniWorld-FourRooms-v0")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = QNet(env.action_space.n).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    returns = []
    successes = 0
    steps = []

    for _ in range(episodes):
        obs, _ = env.reset()
        obs = preprocess_obs(obs)
        done = False
        truncated = False
        ep_ret = 0.0
        ep_steps = 0

        while not (done or truncated):
            with torch.no_grad():
                action = int(torch.argmax(model(torch.from_numpy(obs).unsqueeze(0).to(device)), dim=1).item())
            next_obs, reward, done, truncated, _ = env.step(action)
            obs = preprocess_obs(next_obs)
            ep_ret += reward
            ep_steps += 1

        returns.append(ep_ret)
        steps.append(ep_steps)
        if ep_ret > 0:
            successes += 1

    env.close()
    return float(np.mean(returns)), successes / episodes, float(np.mean(steps))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained RL models on MiniWorld FourRooms")
    parser.add_argument("--algo", choices=["ppo", "dqn", "ddqn"], required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()

    model_path = Path(args.model_path)

    if args.algo == "ppo":
        model = PPO.load(model_path)
        mean_return, success_rate, mean_steps = evaluate_sb3(model, args.episodes)
    elif args.algo == "dqn":
        model = DQN.load(model_path)
        mean_return, success_rate, mean_steps = evaluate_sb3(model, args.episodes)
    else:
        mean_return, success_rate, mean_steps = evaluate_ddqn(model_path, args.episodes)

    print(f"algo: {args.algo}")
    print(f"episodes: {args.episodes}")
    print(f"mean_return: {mean_return:.4f}")
    print(f"success_rate: {success_rate:.4f}")
    print(f"mean_steps: {mean_steps:.2f}")


if __name__ == "__main__":
    main()
