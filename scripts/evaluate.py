"""Evaluate agents on unseen button positions. Run: python -m scripts.evaluate --run runs/<run>"""
import argparse
import csv
import datetime
import os
import warnings
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # for videos, before mujoco loads

import imageio.v2 as imageio
import numpy as np
import yaml
from metaworld.policies import ENV_POLICY_MAP
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
    parser.add_argument("--expert", action="store_true",
                        help="also evaluate Meta-World's hand-written policy (baseline)")
    parser.add_argument("--checkpoint", default="final",
                        help="final, best, untrained, or a step number from checkpoints/")
    # 5 sets of positions that no training run uses
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    parser.add_argument("--episodes-per-seed", type=int, default=10)
    parser.add_argument("--video", type=int, default=0, help="number of videos to record per policy")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions from the policy (like during training) instead of its mean")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ppo.yaml",
                        help="env settings for --random and --expert")
    parser.add_argument("--env", help="task for --random and --expert; default: from --config")
    # noise on the button position (meters)
    parser.add_argument("--noise-jitter", type=float, help="std of a new error at every step, e.g. 0.01")
    parser.add_argument("--noise-bias", type=float, help="std of one error per episode, e.g. 0.01")
    parser.add_argument("--noise-axes", default="xyz", help="which axes get noise, e.g. xy")
    parser.add_argument("--output", type=Path,
                        help="default: results/metrics.csv, or results/robustness.csv with --noise-*")
    return parser.parse_args()


class ExpertPolicy:
    """Meta-World's scripted policy with the same predict() as SB3 models."""

    def __init__(self, env_name: str):
        self.policy = ENV_POLICY_MAP[env_name]()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        with warnings.catch_warnings():  # its actions go outside [-1, 1], env clips them
            warnings.simplefilter("ignore")
            return np.array([self.policy.get_action(o) for o in obs]), None


def noise_kwargs(args: argparse.Namespace) -> dict:
    return {"noise_jitter": args.noise_jitter or 0.0, "noise_bias": args.noise_bias or 0.0,
            "noise_axes": args.noise_axes}


def model_files(run_dir: Path, checkpoint: str) -> tuple[Path, Path]:
    if checkpoint == "final":
        return run_dir / "model.zip", run_dir / "vec_normalize.pkl"
    if checkpoint == "best":
        return run_dir / "best" / "best_model.zip", run_dir / "best" / "vec_normalize.pkl"
    folder = run_dir / "checkpoints"
    return (folder / f"model_{checkpoint}_steps.zip",
            folder / f"model_vecnormalize_{checkpoint}_steps.pkl")


def make_eval_env(config: dict, seed: int, vecnorm_path: Path | None, **env_kwargs):
    """One env on the positions of `seed` with the saved normalization stats."""
    env = make_vec_env(config["env"]["name"], seed=seed, n_envs=1, subprocess=False,
                       reward_function_version=config["env"]["reward_function_version"],
                       reward_shaping=config["env"].get("reward_shaping"), **env_kwargs)
    env.seed(seed)  # same positions every time
    env.action_space.seed(seed)
    if vecnorm_path is None:
        return env
    env = VecNormalize.load(str(vecnorm_path), env)
    env.training = False      # don't change the stats
    env.norm_reward = False   # real rewards in results
    return env


def load_policy(run_dir: Path, checkpoint: str, config: dict) -> tuple[BaseAlgorithm, Path | None]:
    algo = ALGOS[config["run"]["algo"]]
    if checkpoint == "untrained":
        # same config + seed = the same initial weights as in training
        # fresh normalization (mean 0, std 1) doesn't change obs, so we skip it
        env = make_eval_env(config, seed=0, vecnorm_path=None)
        model = algo("MlpPolicy", env, seed=config["run"]["seed"], device="cpu",
                     **config[config["run"]["algo"]])
        env.close()
        return model, None
    model_path, vecnorm_path = model_files(run_dir, checkpoint)
    return algo.load(model_path, device="cpu"), vecnorm_path


def run_episodes(env, model: BaseAlgorithm | None, n_episodes: int, frames: list | None = None,
                 stochastic: bool = False) -> list[dict]:
    """Play n episodes: success, first success step and total reward of each."""
    records = []
    obs = env.reset()
    step, total_reward, first_success = 0, 0.0, None
    while len(records) < n_episodes:
        if frames is not None:
            frames.append(np.flipud(env.get_images()[0]))  # MuJoCo renders upside down
        if model is None:
            action = np.array([env.action_space.sample()])
        else:
            # mean action, or a random sample with --stochastic
            action, _ = model.predict(obs, deterministic=not stochastic)
        obs, reward, done, infos = env.step(action)
        step += 1
        total_reward += float(reward[0])
        if first_success is None and infos[0]["success"]:
            first_success = step
        if done[0]:  # env resets by itself
            records.append({"success": bool(infos[0]["is_success"]),
                            "first_success_step": first_success,
                            "total_reward": total_reward})
            step, total_reward, first_success = 0, 0.0, None
    return records


def summarize(records: list[dict]) -> dict:
    # first success step only over successful episodes
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
    records = []
    for seed in args.eval_seeds:
        env = make_eval_env(config, seed, vecnorm_path, **noise_kwargs(args))
        records += run_episodes(env, model, args.episodes_per_seed, stochastic=args.stochastic)
        env.close()
    return summarize(records)


def record_videos(config: dict, model: BaseAlgorithm | None, vecnorm_path: Path | None,
                  n_videos: int, name: str, seed: int, stochastic: bool, noise: dict) -> None:
    """n episodes as mp4, each with another button position."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    env = make_eval_env(config, seed, vecnorm_path, render_mode="rgb_array", camera_name="corner2", **noise)
    for i in range(1, n_videos + 1):
        frames = []
        run_episodes(env, model, 1, frames, stochastic)
        path = VIDEO_DIR / f"{name}_{i}.mp4"
        imageio.mimsave(path, frames, fps=50)  # 500 steps -> 10 s video
        print(f"  saved {path.relative_to(ROOT)}")
    env.close()


def save_metrics(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    # an existing table keeps its own columns, so old rows never shift
    fields = list(row) if is_new else next(csv.reader(path.open()))
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    # noise tests go to their own table
    robustness = args.noise_jitter is not None or args.noise_bias is not None
    output = args.output or ROOT / "results" / ("robustness.csv" if robustness else "metrics.csv")
    noise = noise_kwargs(args)
    # (name, config, model, vecnorm_path)
    policies = []
    baseline_config = yaml.safe_load(args.config.read_text())
    if args.env:
        baseline_config["env"]["name"] = args.env
    if args.random:
        config = {**baseline_config, "run": {"algo": "random", "seed": ""}}
        policies.append(("random", config, None, None))
    if args.expert:
        config = {**baseline_config, "run": {"algo": "expert", "seed": ""}}
        # expert reads raw coordinates
        policies.append(("expert", config, ExpertPolicy(config["env"]["name"]), None))
    for run_dir in args.run:
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        model, vecnorm_path = load_policy(run_dir, args.checkpoint, config)
        # old runs have no variant
        name = f"{config['run'].get('variant', config['run']['algo'])}_seed{config['run']['seed']}_{args.checkpoint}"
        if args.stochastic:
            name += "_stochastic"
        policies.append((name, config, model, vecnorm_path))

    for name, config, model, vecnorm_path in policies:
        metrics = evaluate(config, model, vecnorm_path, args)
        print(f"{name}: {metrics}")
        noise_columns = noise if robustness else {}
        save_metrics(output, {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "env": config["env"]["name"],
            "policy": name,
            "algo": config["run"]["algo"],
            "variant": config["run"].get("variant", config["run"]["algo"]),
            "seed": config["run"]["seed"],
            "checkpoint": args.checkpoint if isinstance(model, BaseAlgorithm) else "",
            "eval_seeds": " ".join(map(str, args.eval_seeds)),
            **noise_columns,
            **metrics,
        })
        if args.video:
            video_name = f"{config['env']['name']}_{name}" + (f"_jitter{noise['noise_jitter']}_bias{noise['noise_bias']}" if robustness else "")
            record_videos(config, model, vecnorm_path, args.video, video_name, args.eval_seeds[0],
                          args.stochastic, noise)


if __name__ == "__main__":
    main()
