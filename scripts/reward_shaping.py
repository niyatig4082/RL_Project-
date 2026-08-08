from __future__ import annotations

import gymnasium as gym
import numpy as np


class GoalDistanceRewardWrapper(gym.Wrapper):
    """Add dense progress reward based on distance to the goal."""

    def __init__(self, env: gym.Env, progress_scale: float = 0.25, step_penalty: float = 0.001, success_bonus: float = 5.0):
        super().__init__(env)
        self._prev_distance: float | None = None
        self.progress_scale = progress_scale
        self.step_penalty = step_penalty
        self.success_bonus = success_bonus

    def _get_goal_pos(self) -> np.ndarray:
        # MiniWorld FourRooms represents the goal as a red box entity.
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
        return obs, shaped_reward, terminated, truncated, info
