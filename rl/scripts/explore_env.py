"""Explore Meta-World tasks: print observations, run random actions, save renders."""
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import imageio.v2 as imageio
import metaworld  # noqa: F401  (importing registers the Meta-World envs in gymnasium)
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TASKS = ["button-press-v3", "lever-pull-v3"]
CAMERAS = ["corner2", "topview"]


def describe_obs(obs: np.ndarray) -> None:
    """Print the 39-number observation split into its named parts."""
    np.set_printoptions(precision=3, suppress=True)
    print(f"  end effector xyz : {obs[0:3]}")
    print(f"  gripper opening  : {obs[3]:.3f}")
    print(f"  object xyz       : {obs[4:7]}")
    print(f"  object quaternion: {obs[7:11]}")
    print(f"  goal xyz         : {obs[36:39]}")


def save_render(env_name: str, camera: str) -> None:
    """Render one frame of the scene from the given camera and save it as PNG."""
    env = gym.make("Meta-World/MT1", env_name=env_name, seed=0,
                   render_mode="rgb_array", camera_name=camera)
    env.reset()
    # MuJoCo's offscreen render comes out upside down, so flip it vertically
    img = np.flipud(env.render())
    path = RESULTS_DIR / f"{env_name}_{camera}.png"
    imageio.imwrite(path, img)
    print(f"  saved {path.name}")
    env.close()


def show_reset_variation(env_name: str) -> None:
    """Reset several times: the object/goal position should change between resets."""
    env = gym.make("Meta-World/MT1", env_name=env_name, seed=0)
    print("  object xyz after 3 resets:")
    for _ in range(3):
        obs, _ = env.reset()
        print(f"    {np.round(obs[4:7], 3)}")
    env.close()


def probe_axes(env_name: str) -> None:
    """Push the end effector along +x, +y, +z to learn which axis is which."""
    env = gym.make("Meta-World/MT1", env_name=env_name, seed=0)
    for axis, name in enumerate("xyz"):
        obs, _ = env.reset()
        start = obs[0:3].copy()
        action = np.zeros(4, dtype=np.float32)
        action[axis] = 1.0
        for _ in range(10):
            obs, *_ = env.step(action)
        print(f"  action +{name}: end effector moved by {np.round(obs[0:3] - start, 3)}")
    env.close()


def random_rollout(env_name: str, max_steps: int = 600) -> None:
    """Run random actions until the episode ends; report reward, success and termination."""
    env = gym.make("Meta-World/MT1", env_name=env_name, seed=0)
    env.reset()
    total_reward, ever_success = 0.0, False
    for step in range(1, max_steps + 1):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        ever_success |= bool(info.get("success", 0.0))
        if step in (1, 2, 3):
            print(f"  step {step}: reward={reward:.3f} success={info['success']} "
                  f"terminated={terminated} truncated={truncated}")
        if terminated or truncated:
            break
    print(f"  episode ended at step {step} (terminated={terminated}, truncated={truncated}), "
          f"total reward={total_reward:.1f}, success at any point={ever_success}")
    print(f"  info keys: {sorted(info.keys())}")
    env.close()


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    for env_name in TASKS:
        print(f"\n=== {env_name} ===")
        env = gym.make("Meta-World/MT1", env_name=env_name, seed=0)
        obs, _ = env.reset()
        print(f"action space: {env.action_space}")
        print(f"observation shape: {obs.shape}")
        describe_obs(obs)
        env.close()

        show_reset_variation(env_name)
        probe_axes(env_name)
        random_rollout(env_name)
        for camera in CAMERAS:
            save_render(env_name, camera)


if __name__ == "__main__":
    main()
