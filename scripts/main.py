from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO and DQN, evaluate them, and generate summary artifacts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--ppo-steps", type=int, default=200_000)
    parser.add_argument("--dqn-steps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-dir", type=str, default="logs")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.seeds:
        raise ValueError("--seeds must include at least one seed")
    if args.ppo_steps <= 0:
        raise ValueError("--ppo-steps must be > 0")
    if args.dqn_steps <= 0:
        raise ValueError("--dqn-steps must be > 0")
    if args.eval_freq <= 0:
        raise ValueError("--eval-freq must be > 0")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be > 0")
    if args.frame_stack <= 0:
        raise ValueError("--frame-stack must be > 0")


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_and_evaluate(algo: str, seed: int, args: argparse.Namespace, output_dir: Path, log_dir: Path) -> dict[str, float | int | str]:
    if algo == "ppo":
        run_cmd(
            [
                sys.executable,
                "scripts/train_ppo_sb3.py",
                "--seed",
                str(seed),
                "--total-timesteps",
                str(args.ppo_steps),
                "--eval-freq",
                str(args.eval_freq),
                "--eval-episodes",
                str(args.eval_episodes),
                "--frame-stack",
                str(args.frame_stack),
                "--run-name",
                f"seed_{seed}",
            ]
        )
        model_path = output_dir / f"ppo_seed_{seed}"
    else:
        run_cmd(
            [
                sys.executable,
                "scripts/train_dqn_sb3.py",
                "--seed",
                str(seed),
                "--total-timesteps",
                str(args.dqn_steps),
                "--eval-freq",
                str(args.eval_freq),
                "--eval-episodes",
                str(args.eval_episodes),
                "--frame-stack",
                str(args.frame_stack),
                "--run-name",
                f"seed_{seed}",
            ]
        )
        model_path = output_dir / f"dqn_seed_{seed}.zip"

    eval_cmd = [
        sys.executable,
        "scripts/main.py",
        "--eval-only",
        "--algo",
        algo,
        "--model-path",
        str(model_path),
        "--episodes",
        str(args.eval_episodes),
        "--frame-stack",
        str(args.frame_stack),
        "--output-dir",
        str(output_dir),
        "--log-dir",
        str(log_dir),
    ]
    run_cmd(eval_cmd)

    return {"algo": algo, "seed": seed, "model_path": str(model_path)}


def evaluate_model(algo: str, model_path: Path, episodes: int, frame_stack: int, output_dir: Path, log_dir: Path) -> tuple[float, float, float]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if algo == "ppo":
        from stable_baselines3 import PPO
        import miniworld
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

        env = DummyVecEnv([lambda: GoalDistanceRewardWrapper(gym.make("MiniWorld-FourRooms-v0"))])
        env = VecTransposeImage(env)
        if frame_stack > 1:
            env = VecFrameStack(env, n_stack=frame_stack)
        model = PPO.load(model_path, env=env)
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
                ep_ret += float(reward[0])
                ep_steps += 1
            returns.append(ep_ret)
            steps.append(ep_steps)
            if done and ep_ret > 0:
                successes += 1
        env.close()
        return float(sum(returns) / len(returns)), successes / episodes, float(sum(steps) / len(steps))

    from stable_baselines3 import DQN
    import miniworld
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

    env = DummyVecEnv([lambda: GoalDistanceRewardWrapper(gym.make("MiniWorld-FourRooms-v0"))])
    env = VecTransposeImage(env)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
    model = DQN.load(model_path, env=env)
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
            ep_ret += float(reward[0])
            ep_steps += 1
        returns.append(ep_ret)
        steps.append(ep_steps)
        if done and ep_ret > 0:
            successes += 1
    env.close()
    return float(sum(returns) / len(returns)), successes / episodes, float(sum(steps) / len(steps))


def generate_plots(summary_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_path)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    for algo in sorted(df["algo"].unique()):
        subset = df[df["algo"] == algo]
        plt.plot(subset["seed"], subset["mean_return"], marker="o", label=algo)
    plt.xlabel("seed")
    plt.ylabel("mean return")
    plt.title("Mean return by seed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "reward_vs_timesteps.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    summary = df.groupby("algo")["success_rate"].mean()
    summary.plot(kind="bar", color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("success rate")
    plt.title("Mean success rate")
    plt.tight_layout()
    plt.savefig(plots_dir / "final_success_rate.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    summary = df.groupby("algo")["mean_return"].mean()
    summary.plot(kind="bar", color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("mean return")
    plt.title("Mean return")
    plt.tight_layout()
    plt.savefig(plots_dir / "final_mean_return.png")
    plt.close()


def validate_outputs(output_dir: Path) -> None:
    required = [
        output_dir / "multiseed_summary.csv",
        output_dir / "plots" / "reward_vs_timesteps.png",
        output_dir / "plots" / "final_success_rate.png",
        output_dir / "plots" / "final_mean_return.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required outputs: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO and DQN, evaluate them, and generate summary artifacts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--ppo-steps", type=int, default=200_000)
    parser.add_argument("--dqn-steps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    if args.eval_only:
        if args.algo is None or args.model_path is None or args.episodes is None:
            raise ValueError("--eval-only requires --algo, --model-path, and --episodes")
        output_dir = Path(args.output_dir)
        log_dir = Path(args.log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        mean_return, success_rate, mean_steps = evaluate_model(args.algo, Path(args.model_path), args.episodes, args.frame_stack, output_dir, log_dir)
        print(f"algo: {args.algo}")
        print(f"episodes: {args.episodes}")
        print(f"mean_return: {mean_return:.4f}")
        print(f"success_rate: {success_rate:.4f}")
        print(f"mean_steps: {mean_steps:.2f}")
        return

    validate_args(args)

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for algo in ["ppo", "dqn"]:
        for seed in args.seeds:
            result = train_and_evaluate(algo, seed, args, output_dir, log_dir)
            rows.append(result)

    summary_path = output_dir / "multiseed_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "seed", "model_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary to {summary_path}")
    generate_plots(summary_path, output_dir)
    validate_outputs(output_dir)
    print("All requested outputs generated.")


if __name__ == "__main__":
    main()
