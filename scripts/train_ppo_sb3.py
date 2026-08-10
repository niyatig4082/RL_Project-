import argparse
from pathlib import Path

import gymnasium as gym
import miniworld
import torch
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage

from reward_wrappers import StrictGoalRewardWrapper


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
        img = Image.new("RGB", (64, 64), color)
        img.save(target)


def make_env() -> gym.Env:
    ensure_miniworld_textures()
    env = gym.make("MiniWorld-FourRooms-v0", max_episode_steps=250)
    return StrictGoalRewardWrapper(env, max_episode_steps=250)


def select_device(requested: str | None = None) -> str:
    if requested in {None, "auto"}:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested is not None and requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[warn] CUDA was requested but is not available; falling back to CPU")
            return "cpu"
        if ":" in requested:
            device_idx = int(requested.split(":", 1)[1])
            if device_idx >= torch.cuda.device_count():
                print(f"[warn] CUDA device index {device_idx} is not available; falling back to CPU")
                return "cpu"
            torch.cuda.set_device(device_idx)
            return requested
        torch.cuda.set_device(0)
        return "cuda"

    raise ValueError(f"Unsupported device: {requested}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on MiniWorld FourRooms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--allow-training", action="store_true", help="Explicitly allow PPO training to start")
    parser.add_argument(
        "--checkpoint-total-steps",
        type=int,
        default=100_000,
        help="Approximate global step interval for PPO checkpoints.",
    )
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be > 0")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be > 0")
    if args.eval_freq <= 0:
        raise ValueError("--eval-freq must be > 0")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be > 0")
    if args.checkpoint_total_steps <= 0:
        raise ValueError("--checkpoint-total-steps must be > 0")
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
    if not args.allow_training:
        raise PermissionError("Training is disabled by default. Re-run with --allow-training to start PPO training.")
    device = select_device(args.device)
    print(f"[info] PPO using device: {device}")

    run_name = args.run_name or f"seed_{args.seed}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_log_dir = Path(args.log_dir) / "ppo" / run_name
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    def make_train_env(rank: int):
        def _init():
            env = make_env()
            env.reset(seed=args.seed + rank)
            return env

        return _init

    if args.n_envs == 1:
        env = DummyVecEnv([make_train_env(0)])
    else:
        env = SubprocVecEnv([make_train_env(i) for i in range(args.n_envs)])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    eval_env = DummyVecEnv([lambda: Monitor(make_env())])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env.seed(args.seed + 1)

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
        tensorboard_log=str(Path(args.log_dir) / "ppo"),
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

    checkpoint_steps_per_env = max(args.checkpoint_total_steps // args.n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_steps_per_env,
        save_path=str(eval_log_dir / "checkpoints"),
        name_prefix="ppo_fourrooms",
    )

    model.learn(total_timesteps=args.total_timesteps, progress_bar=False, callback=[eval_callback, checkpoint_callback])
    artifact_path = output_dir / f"ppo_{run_name}.zip"
    model.save(artifact_path)
    print(f"[info] saved model to {artifact_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
