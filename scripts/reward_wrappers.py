from __future__ import annotations

import gymnasium as gym


class StrictGoalRewardWrapper(gym.Wrapper):
    """Enforce the sparse reward rule for MiniWorld FourRooms.

    Reward is zero on every normal step. The episode only gets a positive reward
    when the agent reaches the red box, and the reward is computed as:
    1 - 0.2 * (step_count / max_episode_steps)
    """

    def __init__(self, env: gym.Env, *, max_episode_steps: int = 250):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options) if seed is not None else self.env.reset(options=options)
        self._step_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        if hasattr(self.env, "near") and hasattr(self.env, "box"):
            if self.env.near(self.env.box):
                reward = 1.0 - 0.2 * (self._step_count / self.max_episode_steps)
                terminated = True
            else:
                reward = 0.0
        else:
            reward = 0.0

        return obs, reward, terminated, truncated, info
