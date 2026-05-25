"""Train SAC on the fixed-start/fixed-goal 2D point-mass env.

Run from the repo root so the ``src.smerl`` import resolves:

    python -m src.smerl.train_sac --total-timesteps 100000
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.smerl.point2d_env import Point2DGoalEnv


def make_env(seed: int, monitor_dir: str | None = None,
             start: tuple[float, float] | None = None,
             goal: tuple[float, float] | None = None):
    def _thunk():
        env = Point2DGoalEnv(seed=seed, start=start, goal=goal)
        if monitor_dir is not None:
            os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=os.path.join(monitor_dir, "train"))
        return env
    return _thunk


def evaluate(model: SAC, env_seed: int, n_episodes: int = 10,
             start: tuple[float, float] | None = None,
             goal: tuple[float, float] | None = None) -> dict:
    env = Point2DGoalEnv(seed=env_seed, start=start, goal=goal)
    successes, returns, final_dists, lengths = [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret, ep_len = 0.0, 0
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ep_ret += float(r)
            ep_len += 1
        successes.append(bool(info.get("is_success", False)))
        returns.append(ep_ret)
        final_dists.append(float(info.get("distance_to_goal", np.nan)))
        lengths.append(ep_len)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_final_dist": float(np.mean(final_dists)),
        "mean_length": float(np.mean(lengths)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-seed", type=int, default=0,
                        help="seed for the fixed start/goal pair")
    parser.add_argument("--algo-seed", type=int, default=42,
                        help="seed for SAC")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--log-dir", type=str,
                        default="src/smerl/runs/sac_point2d")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu is fine — 2D obs + small MLP")
    parser.add_argument("--start", type=float, nargs=2, default=None,
                        metavar=("X", "Y"),
                        help="explicit start position (overrides --env-seed)")
    parser.add_argument("--goal", type=float, nargs=2, default=None,
                        metavar=("X", "Y"),
                        help="explicit goal position (overrides --env-seed)")
    args = parser.parse_args()

    start = tuple(args.start) if args.start is not None else None
    goal = tuple(args.goal) if args.goal is not None else None

    os.makedirs(args.log_dir, exist_ok=True)
    monitor_dir = os.path.join(args.log_dir, "monitor")

    env = DummyVecEnv([make_env(args.env_seed, monitor_dir=monitor_dir,
                                start=start, goal=goal)])
    eval_env = DummyVecEnv([make_env(args.env_seed, start=start, goal=goal)])

    probe = Point2DGoalEnv(seed=args.env_seed, start=start, goal=goal)
    print(f"[env] start={probe.start.tolist()}  goal={probe.goal.tolist()}  "
          f"distance={float(np.linalg.norm(probe.goal - probe.start)):.3f}")

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.log_dir, "best"),
        log_path=os.path.join(args.log_dir, "eval"),
        eval_freq=2_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=1_000,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=args.algo_seed,
        device=args.device,
        tensorboard_log=os.path.join(args.log_dir, "tb"),
        verbose=1,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=eval_cb,
                progress_bar=False)
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.1f}s")

    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    print(f"[train] saved final model to {final_path}")

    stats = evaluate(model, args.env_seed, n_episodes=20, start=start, goal=goal)
    print(f"[eval] {stats}")

    with open(os.path.join(args.log_dir, "eval_summary.txt"), "w") as f:
        f.write(repr(stats) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
