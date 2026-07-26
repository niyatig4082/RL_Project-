import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


def make_env() -> gym.Env:
    return gym.make("MiniWorld-FourRooms-v0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on MiniWorld FourRooms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_name = args.run_name or f"seed_{args.seed}"
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_log_dir = Path("logs") / "dqn" / run_name
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    eval_env = DummyVecEnv([lambda: Monitor(make_env())])
    eval_env = VecTransposeImage(eval_env)

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="logs/dqn",
        seed=args.seed,
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

    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=eval_callback)
    model.save(output_dir / f"dqn_{run_name}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
