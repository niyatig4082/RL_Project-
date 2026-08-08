import argparse
from pathlib import Path

import gymnasium as gym
import miniworld
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from reward_shaping import GoalDistanceRewardWrapper


def make_env() -> gym.Env:
    return GoalDistanceRewardWrapper(gym.make("MiniWorld-FourRooms-v0"))


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on MiniWorld FourRooms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.25)
    parser.add_argument("--exploration-initial-eps", type=float, default=0.9)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
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
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be > 0")
    if args.learning_starts < 0:
        raise ValueError("--learning-starts must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.train_freq <= 0:
        raise ValueError("--train-freq must be > 0")
    if args.gradient_steps <= 0:
        raise ValueError("--gradient-steps must be > 0")
    if args.target_update_interval <= 0:
        raise ValueError("--target-update-interval must be > 0")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = select_device()
    print(f"[info] DQN using device: {device}")

    run_name = args.run_name or f"seed_{args.seed}"
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_log_dir = Path("logs") / "dqn" / run_name
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

    effective_learning_starts = min(args.learning_starts, max(100, args.total_timesteps // 4))

    if effective_learning_starts != args.learning_starts:
        print(
            "[info] Adjusted learning_starts "
            f"from {args.learning_starts} to {effective_learning_starts} "
            "to ensure updates occur within the training budget."
        )

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=effective_learning_starts,
        batch_size=args.batch_size,
        tau=1.0,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        tensorboard_log="logs/dqn",
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
    model.save(output_dir / f"dqn_{run_name}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
