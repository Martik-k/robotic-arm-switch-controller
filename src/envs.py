"""Factory for wrapped Meta-World environments."""
import gymnasium as gym
import metaworld  # noqa: F401  (importing registers the Meta-World envs in gymnasium)

from src.wrappers import Float32Obs, IsSuccess


def make_env(env_name: str, seed: int, **kwargs) -> gym.Env:
    """Create a Meta-World task wrapped for SB3; `seed` selects the set of object positions."""
    # The passive checker only warns about Meta-World's inexact bounds, which Float32Obs fixes
    env = gym.make(
        "Meta-World/MT1", env_name=env_name, seed=seed, disable_env_checker=True, **kwargs
    )
    return IsSuccess(Float32Obs(env))
