from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env() -> gym.Env:
    return gym.make("MiniWorld-FourRooms-v0")


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env])

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="logs/ppo",
    )

    model.learn(total_timesteps=100_000, progress_bar=True)
    model.save(output_dir / "ppo_fourrooms")

    env.close()


if __name__ == "__main__":
    main()
