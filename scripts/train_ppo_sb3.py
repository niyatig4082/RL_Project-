import argparse
from pathlib import Path

import gymnasium as gym
import miniworld
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage


class GoalDistanceRewardWrapper(gym.Wrapper):
    """Add dense progress reward based on distance to the goal."""

    def __init__(self, env: gym.Env, progress_scale: float = 0.5, step_penalty: float = 0.001, success_bonus: float = 10.0):
        super().__init__(env)
        self._prev_distance: float | None = None
        self.progress_scale = progress_scale
        self.step_penalty = step_penalty
        self.success_bonus = success_bonus

    def _get_goal_pos(self) -> np.ndarray:
        env = self.env.unwrapped
        entities = getattr(env, "entities", None) or []
        for entity in entities:
            color = getattr(entity, "color", None)
            if color == "red":
                return np.asarray(entity.pos, dtype=np.float32)
        raise AttributeError("Environment does not expose a goal position")

    def _get_agent_pos(self) -> np.ndarray:
        agent = getattr(self.env.unwrapped, "agent", None)
        if agent is not None:
            return np.asarray(agent.pos, dtype=np.float32)
        raise AttributeError("Environment does not expose an agent position")

    def _distance_to_goal(self) -> float:
        agent_pos = self._get_agent_pos()
        goal_pos = self._get_goal_pos()
        return float(np.linalg.norm(agent_pos - goal_pos))

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_distance = self._distance_to_goal()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        distance = self._distance_to_goal()
        if self._prev_distance is None:
            shaped_reward = reward
        else:
            progress = self._prev_distance - distance
            shaped_reward = float(reward) + self.progress_scale * progress - self.step_penalty
            if distance < 1.0:
                shaped_reward += self.success_bonus
        self._prev_distance = distance
        info = dict(info or {})
        info["success"] = bool(terminated and distance < 1.0)
        info["distance_to_goal"] = distance
        return obs, shaped_reward, terminated, truncated, info


def make_env() -> gym.Env:
    return GoalDistanceRewardWrapper(gym.make("MiniWorld-FourRooms-v0"))


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on MiniWorld FourRooms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be > 0")
    if args.eval_freq <= 0:
        raise ValueError("--eval-freq must be > 0")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be > 0")
    if args.frame_stack <= 0:
        raise ValueError("--frame-stack must be > 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.n_epochs <= 0:
        raise ValueError("--n-epochs must be > 0")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = select_device()
    print(f"[info] PPO using device: {device}")

    run_name = args.run_name or f"seed_{args.seed}"
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_log_dir = Path("logs") / "ppo" / run_name
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    eval_env = DummyVecEnv([lambda: Monitor(make_env())])
    eval_env = VecTransposeImage(eval_env)
    env.seed(args.seed)
    eval_env.seed(args.seed + 1)

    if args.frame_stack > 1:
        env = VecFrameStack(env, n_stack=args.frame_stack)
        eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log="logs/ppo",
        seed=args.seed,
        device=device,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_log_dir / "best_model"),
        log_path=str(eval_log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.total_timesteps, progress_bar=False, callback=eval_callback)
    model.save(output_dir / f"ppo_{run_name}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
