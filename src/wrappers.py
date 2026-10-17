"""Gymnasium wrappers that adapt Meta-World environments for Stable-Baselines3."""
import gymnasium as gym
import numpy as np


class Float32Obs(gym.ObservationWrapper):
    """Cast observations to float32 and declare an unbounded observation space."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)


class IsSuccess(gym.Wrapper):
    """Set `info["is_success"]` to whether success happened at any step of the episode."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._succeeded = False

    def reset(self, **kwargs):
        self._succeeded = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._succeeded |= bool(info.get("success", 0.0))
        info["is_success"] = self._succeeded
        return obs, reward, terminated, truncated, info
