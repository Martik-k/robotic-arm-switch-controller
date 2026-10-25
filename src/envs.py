"""Factory for wrapped Meta-World environments."""
from pathlib import Path

import gymnasium as gym
import metaworld  # noqa: F401  (importing registers the Meta-World envs in gymnasium)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from src.wrappers import AdditiveReward, Float32Obs, IsSuccess, LeverAngleSuccess, NoisyObjectPosition


def make_env(env_name: str, seed: int, noise_jitter: float = 0.0, noise_bias: float = 0.0,
             noise_axes: str = "xyz", reward_shaping: str | None = None, **kwargs) -> gym.Env:
    """Create a Meta-World task wrapped for SB3; `seed` selects the set of object positions."""
    # The passive checker only warns about Meta-World's inexact bounds, which Float32Obs fixes
    env = gym.make(
        "Meta-World/MT1", env_name=env_name, seed=seed, disable_env_checker=True, **kwargs
    )
    env = Float32Obs(env)
    if noise_jitter or noise_bias:  # meters, see NoisyObjectPosition
        env = NoisyObjectPosition(env, jitter=noise_jitter, bias=noise_bias, axes=noise_axes)
    if reward_shaping == "additive":  # see AdditiveReward
        env = AdditiveReward(env)
    if env_name == "lever-pull-v3":  # same success rule for every reward version
        env = LeverAngleSuccess(env)
    return IsSuccess(env)


def make_vec_env(
    env_name: str,
    seed: int,
    n_envs: int,
    subprocess: bool = True,
    monitor_dir: Path | None = None,
    **kwargs,
) -> VecEnv:
    """n_envs copies in Monitor; copy i uses seed + i, i.e. its own set of positions."""
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)

    def env_fn(i: int):
        def init() -> gym.Env:
            filename = None if monitor_dir is None else str(monitor_dir / str(i))
            # so Monitor also logs is_success
            return Monitor(make_env(env_name, seed=seed + i, **kwargs),
                           filename=filename, info_keywords=("is_success",))
        return init

    env_fns = [env_fn(i) for i in range(n_envs)]
    # subprocesses = copies run on different CPU cores
    return SubprocVecEnv(env_fns) if subprocess else DummyVecEnv(env_fns)
