"""Training script for Meta-World tasks. Run: python -m scripts.train"""
import argparse
import datetime
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from src.envs import make_vec_env

ROOT = Path(__file__).resolve().parent.parent
ALGOS = {"ppo": PPO, "sac": SAC}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--seed", type=int, default=0)
    # if an argument is not given (None), load_config takes the value from the config
    parser.add_argument("--config", type=Path, help="default: configs/<algo>.yaml")
    parser.add_argument("--env", help="default: from config")
    parser.add_argument("--total-steps", type=int, help="default: from config")
    parser.add_argument("--n-envs", type=int, help="default: from config, at most CPU cores")
    parser.add_argument("--device", help="default: cpu for PPO, auto for SAC")
    parser.add_argument("--run-name", help="default: <algo>_<env>_seed<N>_<timestamp>")
    parser.add_argument("--monitor-csv", action=argparse.BooleanOptionalAction,
                        help="save per-episode CSV (--no-monitor-csv to disable); default: from config")
    parser.add_argument("--keep-checkpoints", type=int,
                        help="keep only the N newest checkpoints; default: from config (all)")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction,
                        help="print a progress table every rollout (--no-verbose to disable); "
                             "default: from config")
    return parser.parse_args()


def cpu_name() -> str:
    """Get the CPU name for the report (/proc/cpuinfo is only on Linux)."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def load_config(args: argparse.Namespace) -> dict:
    """Load the yaml config, replace values with the ones from the command line, add run info."""
    path = args.config or ROOT / "configs" / f"{args.algo}.yaml"
    config = yaml.safe_load(path.read_text())

    if args.env is not None:
        config["env"]["name"] = args.env
    if args.total_steps is not None:
        config["training"]["total_steps"] = args.total_steps
    if args.n_envs is not None:
        config["env"]["n_envs"] = args.n_envs
    else:
        # more envs than cores = they just wait for each other, no real speedup
        config["env"]["n_envs"] = min(config["env"]["n_envs"], os.cpu_count())
    if args.monitor_csv is not None:
        config["training"]["save_monitor_csv"] = args.monitor_csv
    if args.keep_checkpoints is not None:
        config["training"]["keep_checkpoints"] = args.keep_checkpoints
    if args.verbose is not None:
        config["training"]["verbose"] = args.verbose

    config["run"] = {
        "algo": args.algo,
        "seed": args.seed,
        # PPO network is small so GPU doesn't help; SAC trains every step so GPU can help
        "device": args.device or ("cpu" if args.algo == "ppo" else "auto"),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "cpu": cpu_name(),
        "cpu_cores": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return config


def make_run_dir(args: argparse.Namespace, config: dict) -> Path:
    """Make a new folder for the run (never overwrite old ones) and save the config there."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = args.run_name or f"{args.algo}_{config['env']['name']}_seed{args.seed}_{timestamp}"
    run_dir = ROOT / "runs" / name
    suffix = 2
    while run_dir.exists():  # folder already exists -> add _2, _3, ...
        run_dir = ROOT / "runs" / f"{name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return run_dir


def build_envs(config: dict, run_dir: Path) -> tuple[VecNormalize, VecNormalize]:
    """Make the training envs (copy i gets seed + i) and one exam env with new button positions."""
    env_cfg, train_cfg, run = config["env"], config["training"], config["run"]
    norm = config["normalize"]
    env_kwargs = {"reward_function_version": env_cfg["reward_function_version"]}

    train_env = make_vec_env(
        env_cfg["name"], seed=run["seed"], n_envs=env_cfg["n_envs"],
        subprocess=run["algo"] == "ppo",  # SAC uses only one env
        monitor_dir=run_dir / "monitor" if train_cfg["save_monitor_csv"] else None,
        **env_kwargs,
    )
    # same gamma as the algorithm, because reward normalization uses it
    train_env = VecNormalize(train_env, norm_obs=norm["norm_obs"], norm_reward=norm["norm_reward"],
                             gamma=config[run["algo"]]["gamma"])

    eval_env = make_vec_env(env_cfg["name"], seed=train_cfg["eval_seed"], n_envs=1,
                            subprocess=False, **env_kwargs)
    # exam env doesn't update the stats (it gets them from the training env) and keeps real rewards
    eval_env = VecNormalize(eval_env, training=False, norm_obs=norm["norm_obs"], norm_reward=False)
    return train_env, eval_env


def build_model(config: dict, train_env: VecNormalize, run_dir: Path) -> BaseAlgorithm:
    """Create the PPO/SAC agent with parameters from the config."""
    run = config["run"]
    return ALGOS[run["algo"]](
        "MlpPolicy", train_env,
        seed=run["seed"],  # this also sets seeds for python, numpy, torch and the envs
        device=run["device"],
        tensorboard_log=str(run_dir / "tb"),
        verbose=int(config["training"]["verbose"]),
        **config[run["algo"]],
    )


class SaveVecNormalizeCallback(BaseCallback):
    """Save vec_normalize.pkl next to best_model.zip, the model does not work without it."""

    def __init__(self, save_dir: Path):
        super().__init__()
        self.save_dir = save_dir

    def _on_step(self) -> bool:
        self.model.get_vec_normalize_env().save(str(self.save_dir / "vec_normalize.pkl"))
        return True


class PrunedCheckpointCallback(CheckpointCallback):
    """Same as CheckpointCallback, but can delete old checkpoints and keep only the last N."""

    def __init__(self, keep_last: int | None, **kwargs):
        super().__init__(**kwargs)
        self.keep_last = keep_last

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.keep_last and self.n_calls % self.save_freq == 0:
            # names look like model_vecnormalize_100000_steps.pkl, the number is the step
            for pattern in ("model_*_steps.zip", "model_vecnormalize_*_steps.pkl"):
                files = sorted(Path(self.save_path).glob(pattern),
                               key=lambda p: int(p.stem.split("_")[-2]))
                for old in files[:-self.keep_last]:
                    old.unlink()
        return result


def build_callbacks(config: dict, eval_env: VecNormalize, run_dir: Path) -> list[BaseCallback]:
    """Exam every eval_freq steps and save a checkpoint every checkpoint_freq steps."""
    train_cfg, n_envs = config["training"], config["env"]["n_envs"]
    verbose = int(train_cfg["verbose"])
    # callbacks count env.step() calls, one call = n_envs steps, so we divide
    eval_calls = max(train_cfg["eval_freq"] // n_envs, 1)
    checkpoint_calls = max(train_cfg["checkpoint_freq"] // n_envs, 1)

    best_dir = run_dir / "best"
    best_dir.mkdir()
    exam = EvalCallback(
        eval_env,
        n_eval_episodes=train_cfg["n_eval_episodes"],
        eval_freq=eval_calls,
        deterministic=True,  # no random noise during the exam
        log_path=str(run_dir / "eval"),  # rewards and successes of all exams go to evaluations.npz
        best_model_save_path=str(best_dir),
        callback_on_new_best=SaveVecNormalizeCallback(best_dir),
        verbose=verbose,
    )
    checkpoint = PrunedCheckpointCallback(
        keep_last=train_cfg["keep_checkpoints"],
        save_freq=checkpoint_calls,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="model",
        save_vecnormalize=True,  # checkpoint is useless without the normalization stats
        verbose=verbose,
    )
    return [exam, checkpoint]


def write_summary(run_dir: Path, model: BaseAlgorithm, status: str, start_time: float) -> dict:
    """Write how the run went (status, time, last exam result) to summary.yaml."""
    duration = time.time() - start_time
    summary = {
        "status": status,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "duration_hours": round(duration / 3600, 3),
        "env_steps": int(model.num_timesteps),
        "fps": int(model.num_timesteps / duration),
    }
    evaluations = run_dir / "eval" / "evaluations.npz"
    if evaluations.exists():  # no file if we stopped before the first exam
        data = np.load(evaluations)
        summary["last_exam"] = {
            "env_steps": int(data["timesteps"][-1]),
            "success_rate": float(data["successes"][-1].mean()),
            "mean_reward": float(data["results"][-1].mean()),
        }
    (run_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args)
    run_dir = make_run_dir(args, config)
    print(f"run dir: {run_dir}")
    train_env, eval_env = build_envs(config, run_dir)
    model = build_model(config, train_env, run_dir)
    callbacks = build_callbacks(config, eval_env, run_dir)

    start_time = time.time()
    status = "crashed"
    try:
        model.learn(total_timesteps=config["training"]["total_steps"], callback=callbacks)
        status = "finished"
    except KeyboardInterrupt:
        status = "interrupted"
        print("\ninterrupted by Ctrl+C, saving the current model")
    finally:
        # always save the model together with vec_normalize.pkl
        model.save(run_dir / "model.zip")
        train_env.save(str(run_dir / "vec_normalize.pkl"))
        summary = write_summary(run_dir, model, status, start_time)
        for env in (train_env, eval_env):
            try:
                env.close()
            except (EOFError, BrokenPipeError):
                pass  # after Ctrl+C the subprocesses can already be closed

    print(yaml.safe_dump(summary, sort_keys=False))
    print(f"evaluate with: python -m scripts.evaluate --run {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
