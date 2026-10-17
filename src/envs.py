"""Factory for wrapped Meta-World environments."""
from pathlib import Path

import gymnasium as gym
import metaworld  # noqa: F401  (importing registers the Meta-World envs in gymnasium)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from src.wrappers import Float32Obs, IsSuccess


def make_env(env_name: str, seed: int, **kwargs) -> gym.Env:
    """Create a Meta-World task wrapped for SB3; `seed` selects the set of object positions."""
    # The passive checker only warns about Meta-World's inexact bounds, which Float32Obs fixes
    env = gym.make(
        "Meta-World/MT1", env_name=env_name, seed=seed, disable_env_checker=True, **kwargs
    )
    return IsSuccess(Float32Obs(env))


def make_vec_env(
    env_name: str,
    seed: int,
    n_envs: int,
    subprocess: bool = True,
    monitor_dir: Path | None = None,
    **kwargs,
) -> VecEnv:
    """Stack n_envs Monitor-wrapped copies; copy i uses seed + i, i.e. its own set of positions."""
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)

    def env_fn(i: int):
        def init() -> gym.Env:
            filename = None if monitor_dir is None else str(monitor_dir / str(i))
            # info_keywords makes Monitor log is_success at the end of every episode
            return Monitor(make_env(env_name, seed=seed + i, **kwargs),
                           filename=filename, info_keywords=("is_success",))
        return init

    env_fns = [env_fn(i) for i in range(n_envs)]
    # Subprocesses run the copies on separate CPU cores; one copy needs no extra process
    return SubprocVecEnv(env_fns) if subprocess else DummyVecEnv(env_fns)
