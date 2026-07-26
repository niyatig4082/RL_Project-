import gymnasium as gym
import miniworld
import stable_baselines3
import torch


def main() -> None:
    print(f"gymnasium: {gym.__version__}")
    print(f"miniworld: {miniworld.__version__}")
    print(f"stable_baselines3: {stable_baselines3.__version__}")
    print(f"torch: {torch.__version__}")

    env = gym.make("MiniWorld-FourRooms-v0")
    obs, _ = env.reset()

    print(f"observation shape: {obs.shape}")
    print(f"action space: {env.action_space}")
    print("smoke test passed")

    env.close()


if __name__ == "__main__":
    main()
