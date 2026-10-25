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


class NoisyObjectPosition(gym.ObservationWrapper):
    """Noise on the object position the agent sees, like YOLO errors; the sim itself is untouched.

    jitter: new error every step, bias: one error per episode.
    """

    OBJECT, OBJECT_PREV, GOAL = slice(4, 7), slice(22, 25), slice(36, 39)

    def __init__(self, env: gym.Env, jitter: float = 0.0, bias: float = 0.0):
        super().__init__(env)
        self.jitter, self.bias = jitter, bias
        self._rng = np.random.default_rng()
        self._offset = np.zeros(3)
        self._prev_error = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:  # same seed -> same noise
            self._rng = np.random.default_rng(seed)
        self._offset = self._rng.normal(0.0, self.bias, 3)
        self._prev_error = None
        return super().reset(seed=seed, options=options)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        error = self._offset + self._rng.normal(0.0, self.jitter, 3)
        # obs[18:36] is the previous frame, so it gets the previous error
        prev_error = error if self._prev_error is None else self._prev_error
        self._prev_error = error
        obs = obs.copy()
        obs[self.OBJECT] += error
        obs[self.GOAL] += error  # goal x, z = object x, z, otherwise the agent sees the truth there
        obs[self.OBJECT_PREV] += prev_error
        return obs


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
