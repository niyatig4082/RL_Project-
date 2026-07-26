from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class QNet(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Input is expected as uint8 image in channel-first format: [B, 3, 60, 80]
        x = obs.float() / 255.0
        x = self.features(x)
        return self.head(x)


@dataclass
class Config:
    total_steps: int = 200_000
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    train_freq: int = 4
    target_update_freq: int = 5_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.2
    eval_every: int = 20_000
    seed: int = 42


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    # Convert HWC -> CHW for convolutional network consumption
    return np.transpose(obs, (2, 0, 1)).astype(np.uint8)


def epsilon_by_step(step: int, cfg: Config) -> float:
    decay_steps = int(cfg.total_steps * cfg.epsilon_decay_fraction)
    if step >= decay_steps:
        return cfg.epsilon_end
    ratio = step / max(1, decay_steps)
    return cfg.epsilon_start + ratio * (cfg.epsilon_end - cfg.epsilon_start)


def evaluate_policy(env_id: str, q_net: QNet, device: torch.device, episodes: int = 20) -> tuple[float, float, float]:
    env = gym.make(env_id)
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
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                action = int(torch.argmax(q_net(obs_t), dim=1).item())

            next_obs, reward, done, truncated, _ = env.step(action)
            ep_ret += reward
            ep_steps += 1
            obs = preprocess_obs(next_obs)

        returns.append(ep_ret)
        steps.append(ep_steps)
        if ep_ret > 0:
            successes += 1

    env.close()
    return float(np.mean(returns)), successes / episodes, float(np.mean(steps))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Double DQN on MiniWorld FourRooms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        seed=args.seed,
        total_steps=args.total_steps,
        eval_every=args.eval_every,
    )
    env_id = "MiniWorld-FourRooms-v0"
    run_name = args.run_name or f"seed_{cfg.seed}"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    env = gym.make(env_id)
    obs, _ = env.reset(seed=cfg.seed)
    obs = preprocess_obs(obs)

    action_dim = env.action_space.n

    q_online = QNet(action_dim).to(device)
    q_target = QNet(action_dim).to(device)
    q_target.load_state_dict(q_online.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q_online.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.buffer_size)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    rewards_log: List[tuple[int, float, float, float]] = []

    episode_reward = 0.0
    progress = trange(1, cfg.total_steps + 1, desc="DDQN training")

    for step in progress:
        epsilon = epsilon_by_step(step, cfg)

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                action = int(torch.argmax(q_online(obs_t), dim=1).item())

        next_obs, reward, done, truncated, _ = env.step(action)
        next_obs = preprocess_obs(next_obs)
        terminal = bool(done or truncated)

        replay.add((obs, action, reward, next_obs, terminal))
        obs = next_obs
        episode_reward += reward

        if terminal:
            obs, _ = env.reset()
            obs = preprocess_obs(obs)
            episode_reward = 0.0

        if step >= cfg.learning_starts and step % cfg.train_freq == 0 and len(replay) >= cfg.batch_size:
            batch = replay.sample(cfg.batch_size)
            obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)

            obs_t = torch.from_numpy(np.stack(obs_b)).to(device)
            act_t = torch.tensor(act_b, dtype=torch.long, device=device).unsqueeze(1)
            rew_t = torch.tensor(rew_b, dtype=torch.float32, device=device).unsqueeze(1)
            next_obs_t = torch.from_numpy(np.stack(next_obs_b)).to(device)
            done_t = torch.tensor(done_b, dtype=torch.float32, device=device).unsqueeze(1)

            q_values = q_online(obs_t).gather(1, act_t)

            with torch.no_grad():
                # Double DQN target: action selection by online net, evaluation by target net
                next_actions = torch.argmax(q_online(next_obs_t), dim=1, keepdim=True)
                next_q = q_target(next_obs_t).gather(1, next_actions)
                targets = rew_t + cfg.gamma * (1.0 - done_t) * next_q

            loss = nn.functional.smooth_l1_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_online.parameters(), max_norm=10.0)
            optimizer.step()

        if step % cfg.target_update_freq == 0:
            q_target.load_state_dict(q_online.state_dict())

        if step % cfg.eval_every == 0:
            mean_ret, success_rate, mean_steps = evaluate_policy(
                env_id,
                q_online,
                device,
                episodes=args.eval_episodes,
            )
            rewards_log.append((step, mean_ret, success_rate, mean_steps))
            progress.set_postfix(
                eps=f"{epsilon:.3f}",
                eval_return=f"{mean_ret:.3f}",
                success=f"{success_rate:.2f}",
            )

    env.close()

    torch.save(q_online.state_dict(), output_dir / f"ddqn_{run_name}.pt")

    if rewards_log:
        np.savetxt(
            output_dir / "ddqn_eval_log.csv",
            np.array(rewards_log),
            delimiter=",",
            header="step,mean_return,success_rate,mean_steps",
            comments="",
        )

        np.savetxt(
            output_dir / f"ddqn_{run_name}_eval_log.csv",
            np.array(rewards_log),
            delimiter=",",
            header="step,mean_return,success_rate,mean_steps",
            comments="",
        )


if __name__ == "__main__":
    main()
