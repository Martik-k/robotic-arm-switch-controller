"""Evaluate trained agents on unseen button positions. Run: python -m scripts.evaluate --run runs/<run>"""
import argparse
import csv
import datetime
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering for videos, must be set before mujoco loads

import imageio.v2 as imageio
import numpy as np
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from src.envs import make_vec_env

ROOT = Path(__file__).resolve().parent.parent
ALGOS = {"ppo": PPO, "sac": SAC}
VIDEO_DIR = ROOT / "results" / "videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", default=[], help="one or more run folders")
    parser.add_argument("--random", action="store_true", help="also evaluate a random policy (baseline)")
    parser.add_argument("--checkpoint", default="final",
                        help="final, best, untrained, or a step number from checkpoints/")
    # 5 sets of 50 positions that no training run uses (training seeds are seed + i, far below 1000)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    parser.add_argument("--episodes-per-seed", type=int, default=10)
    parser.add_argument("--video", type=int, default=0, help="number of videos to record per policy")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions from the policy (like during training) instead of its mean")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ppo.yaml",
                        help="env settings for --random")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "metrics.csv")
    return parser.parse_args()


def model_files(run_dir: Path, checkpoint: str) -> tuple[Path, Path]:
    """Paths to the model and its vec_normalize.pkl, they always go together."""
    if checkpoint == "final":
        return run_dir / "model.zip", run_dir / "vec_normalize.pkl"
    if checkpoint == "best":
        return run_dir / "best" / "best_model.zip", run_dir / "best" / "vec_normalize.pkl"
    folder = run_dir / "checkpoints"
    return (folder / f"model_{checkpoint}_steps.zip",
            folder / f"model_vecnormalize_{checkpoint}_steps.pkl")


def make_eval_env(config: dict, seed: int, vecnorm_path: Path | None, **env_kwargs):
    """One env on the positions of `seed`, with the normalization stats saved during training."""
    env = make_vec_env(config["env"]["name"], seed=seed, n_envs=1, subprocess=False,
                       reward_function_version=config["env"]["reward_function_version"], **env_kwargs)
    env.seed(seed)  # same order of positions every time we evaluate
    env.action_space.seed(seed)  # same random actions for the random policy
    if vecnorm_path is None:  # random or untrained policy, see load_policy
        return env
    env = VecNormalize.load(str(vecnorm_path), env)
    env.training = False      # don't change the stats the model was trained with
    env.norm_reward = False   # we want real rewards in the results
    return env


def load_policy(run_dir: Path, checkpoint: str, config: dict) -> tuple[BaseAlgorithm, Path | None]:
    """Load the model (or rebuild the untrained one) and the path to its normalization stats."""
    algo = ALGOS[config["run"]["algo"]]
    if checkpoint == "untrained":
        # Same config + same seed = the exact initial weights this run started from.
        # Fresh normalization stats are mean 0, std 1, i.e. they don't change the obs, so we skip them
        env = make_eval_env(config, seed=0, vecnorm_path=None)
        model = algo("MlpPolicy", env, seed=config["run"]["seed"], device="cpu",
                     **config[config["run"]["algo"]])
        env.close()
        return model, None
    model_path, vecnorm_path = model_files(run_dir, checkpoint)
    return algo.load(model_path, device="cpu"), vecnorm_path


def run_episodes(env, model: BaseAlgorithm | None, n_episodes: int, frames: list | None = None,
                 stochastic: bool = False) -> list[dict]:
    """Play n episodes; for each one remember success, first success step and total reward."""
    records = []
    obs = env.reset()
    step, total_reward, first_success = 0, 0.0, None
    while len(records) < n_episodes:
        if frames is not None:
            frames.append(np.flipud(env.get_images()[0]))  # MuJoCo renders upside down
        if model is None:
            action = np.array([env.action_space.sample()])
        else:
            # mean of the action distribution, or a random sample from it with --stochastic
            action, _ = model.predict(obs, deterministic=not stochastic)
        obs, reward, done, infos = env.step(action)
        step += 1
        total_reward += float(reward[0])
        if first_success is None and infos[0]["success"]:
            first_success = step
        if done[0]:  # the env resets by itself, so we just start counting a new episode
            records.append({"success": bool(infos[0]["is_success"]),
                            "first_success_step": first_success,
                            "total_reward": total_reward})
            step, total_reward, first_success = 0, 0.0, None
    return records


def summarize(records: list[dict]) -> dict:
    """The three metrics: success rate, mean first success step (only successful), mean reward."""
    successes = [r["success"] for r in records]
    steps = [r["first_success_step"] for r in records if r["first_success_step"] is not None]
    rewards = [r["total_reward"] for r in records]
    return {
        "episodes": len(records),
        "success_rate": round(float(np.mean(successes)), 4),
        "first_success_step": round(float(np.mean(steps)), 1) if steps else None,
        "reward_mean": round(float(np.mean(rewards)), 1),
        "reward_std": round(float(np.std(rewards)), 1),
    }


def evaluate(config: dict, model: BaseAlgorithm | None, vecnorm_path: Path | None,
             args: argparse.Namespace) -> dict:
    """Run the policy on every eval seed and compute the metrics over all episodes."""
    records = []
    for seed in args.eval_seeds:
        env = make_eval_env(config, seed, vecnorm_path)
        records += run_episodes(env, model, args.episodes_per_seed, stochastic=args.stochastic)
        env.close()
    return summarize(records)


def record_videos(config: dict, model: BaseAlgorithm | None, vecnorm_path: Path | None,
                  n_videos: int, name: str, seed: int, stochastic: bool) -> None:
    """Save n episodes as mp4 (side camera), each one with a different button position."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    env = make_eval_env(config, seed, vecnorm_path, render_mode="rgb_array", camera_name="corner2")
    for i in range(1, n_videos + 1):
        frames = []
        run_episodes(env, model, 1, frames, stochastic)
        path = VIDEO_DIR / f"{name}_{i}.mp4"
        imageio.mimsave(path, frames, fps=50)  # 500 steps -> 10 s video
        print(f"  saved {path.relative_to(ROOT)}")
    env.close()


def save_metrics(path: Path, row: dict) -> None:
    """Append one row to the metrics table (header is written only for a new file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    # (name, config, model, vecnorm_path) for every policy we evaluate
    policies = []
    if args.random:
        config = yaml.safe_load(args.config.read_text())
        config["run"] = {"algo": "random", "seed": ""}
        policies.append(("random", config, None, None))
    for run_dir in args.run:
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        model, vecnorm_path = load_policy(run_dir, args.checkpoint, config)
        name = f"{config['run']['algo']}_seed{config['run']['seed']}_{args.checkpoint}"
        if args.stochastic:  # keep these rows and videos apart from the normal (deterministic) ones
            name += "_stochastic"
        policies.append((name, config, model, vecnorm_path))

    for name, config, model, vecnorm_path in policies:
        metrics = evaluate(config, model, vecnorm_path, args)
        print(f"{name}: {metrics}")
        save_metrics(args.output, {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "policy": name,
            "algo": config["run"]["algo"],
            "seed": config["run"]["seed"],
            "checkpoint": "" if model is None else args.checkpoint,
            "eval_seeds": " ".join(map(str, args.eval_seeds)),
            **metrics,
        })
        if args.video:
            record_videos(config, model, vecnorm_path, args.video, name, args.eval_seeds[0],
                          args.stochastic)


if __name__ == "__main__":
    main()
