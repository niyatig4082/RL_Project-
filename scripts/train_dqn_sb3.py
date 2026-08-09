import argparse
from pathlib import Path

import gymnasium as gym
import miniworld
import torch
from PIL import Image
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage


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
    return gym.make("MiniWorld-FourRooms-v0", max_episode_steps=250)


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
    parser = argparse.ArgumentParser(description="Train DQN on MiniWorld FourRooms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--allow-training", action="store_true", help="Explicitly allow DQN training to start")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.4)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--net-arch", nargs="+", type=int, default=[128, 128])
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be > 0")
    if args.eval_freq <= 0:
        raise ValueError("--eval-freq must be > 0")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be > 0")
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
    if not args.allow_training:
        raise PermissionError("Training is disabled by default. Re-run with --allow-training to start DQN training.")
    device = select_device(args.device)
    print(f"[info] DQN using device: {device}")

    run_name = args.run_name or f"seed_{args.seed}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_log_dir = Path(args.log_dir) / "dqn" / run_name
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    def make_plain_env() -> gym.Env:
        return make_env()

    env = DummyVecEnv([make_plain_env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    eval_env = DummyVecEnv([lambda: Monitor(make_plain_env())])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    env.seed(args.seed)
    eval_env.seed(args.seed + 1)

    effective_learning_starts = min(args.learning_starts, max(100, args.total_timesteps // 10))

    if effective_learning_starts != args.learning_starts:
        print(
            "[info] Adjusted learning_starts "
            f"from {args.learning_starts} to {effective_learning_starts} "
            "to ensure updates occur within the training budget."
        )

    policy_kwargs = dict(net_arch=list(args.net_arch))

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=effective_learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(Path(args.log_dir) / "dqn"),
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
    artifact_path = output_dir / f"dqn_{run_name}.zip"
    model.save(artifact_path)
    print(f"[info] saved model to {artifact_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
