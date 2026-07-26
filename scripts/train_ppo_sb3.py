import argparse
from pathlib import Path

import gymnasium as gym
import miniworld
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage


def make_env() -> gym.Env:
    return gym.make("MiniWorld-FourRooms-v0")


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


def main() -> None:
    args = parse_args()

    run_name = args.run_name or f"seed_{args.seed}"
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_log_dir = Path("logs") / "ppo" / run_name
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    eval_env = DummyVecEnv([lambda: Monitor(make_env())])
    eval_env = VecTransposeImage(eval_env)

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
    model.save(output_dir / f"ppo_{run_name}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
