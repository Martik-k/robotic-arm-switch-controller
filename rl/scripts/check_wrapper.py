"""Check the wrappers: float32 obs, is_success in info, no warnings (run with `python -m scripts.check_wrapper`)."""
import warnings

from src.envs import make_env

ENV_NAME = "button-press-v3"


def main() -> None:
    # Any warning (e.g. "obs is not within the observation space") becomes an error
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env = make_env(ENV_NAME, seed=0)
        obs, _ = env.reset()
        print(f"obs dtype={obs.dtype}, shape={obs.shape}")
        print(f"observation space dtype: {env.observation_space.dtype}")

        # A random episode must end with is_success present in the last info
        truncated = False
        while not truncated:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"last info has is_success: {'is_success' in info} (value={info['is_success']})")
    print("no warnings raised")
    env.close()


if __name__ == "__main__":
    main()
