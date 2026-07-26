from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


def make_env() -> gym.Env:
    return gym.make("MiniWorld-FourRooms-v0")


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

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
    )

    model.learn(total_timesteps=200_000, progress_bar=True)
    model.save(output_dir / "dqn_fourrooms")

    env.close()


if __name__ == "__main__":
    main()
