#!/usr/bin/env python3
"""Collect grouped 2D point-mass trajectories that track cubic splines."""

import argparse
import math
import os
from pathlib import Path

# Set before importing dm_control so backend selection is headless-safe.
os.environ.setdefault("MUJOCO_GL", "egl")

from dm_control import suite
from dm_control.utils.cubic_spline import generate_spline_noise_traj_2d
from dm_control.utils.cubic_spline import scale_to_accel_limit_2d
from dm_control.utils.ff_pd_controller import ff_pd_action_2d
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect point-mass demos using two-pass feasible references."
    )
    parser.add_argument(
        "--num-trajs",
        type=int,
        default=10,
        help="Number of spline groups (one sampled spline per group).",
    )
    parser.add_argument(
        "--num-rollouts-per-spline",
        type=int,
        default=4,
        help="Number of rollouts collected for each spline.",
    )
    parser.add_argument("--task", choices=["easy", "hard"], default="easy")
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=5.0,
        help="Episode timeout for the point-mass environment in seconds.",
    )
    parser.add_argument("--horizon-sec", type=float, default=5.0)
    parser.add_argument("--n-anchors", type=int, default=4)
    parser.add_argument("--velocity-scale", type=float, default=0.15)
    parser.add_argument("--amax", type=float, default=0.15)
    parser.add_argument("--kp", type=float, default=20.0)
    parser.add_argument("--kd", type=float, default=8.0)
    parser.add_argument("--mass", type=float, default=0.3)
    parser.add_argument("--gear", type=float, default=0.1)
    parser.add_argument(
        "--action-noise-var",
        type=float,
        default=0.01,
        help="Variance of zero-mean Gaussian noise added to actions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-path",
        default="datasets/point_mass_spline_episodes.pt",
        help="Output .pt file path",
    )
    return parser.parse_args()


def collect_one_episode(env, rng, args, pr, vr, ar):
    time_step = env.reset()

    # Force each rollout to start at the spline's initial state.
    with env.physics.reset_context():
        env.physics.named.data.qpos["root_x"] = float(pr[0, 0])
        env.physics.named.data.qpos["root_y"] = float(pr[0, 1])
        env.physics.named.data.qvel["root_x"] = float(vr[0, 0])
        env.physics.named.data.qvel["root_y"] = float(vr[0, 1])

    obs_position = []
    obs_velocity = []
    actions = []

    for k in range(pr.shape[0]):
        # k = k % pr.shape[0]
        p = np.asarray(env.physics.position(), dtype=np.float32)
        v = np.asarray(env.physics.velocity(), dtype=np.float32)

        action = ff_pd_action_2d(
            obs_position=p,
            obs_velocity=v,
            pr=pr[k],
            vr=vr[k],
            ar=ar[k],
            Kp=args.kp,
            Kd=args.kd,
            mass=args.mass,
            gear=args.gear,
            control_noise_std=args.action_noise_std,
            rng=rng,
        ).astype(np.float32)

        obs_position.append(p)
        obs_velocity.append(v)
        actions.append(action)

        time_step = env.step(action)
        if time_step.last():
            break

    obs_dict = {
        "position": torch.from_numpy(np.stack(obs_position, axis=0)),
        "velocity": torch.from_numpy(np.stack(obs_velocity, axis=0)),
    }
    actions_tensor = torch.from_numpy(np.stack(actions, axis=0))
    length = int(actions_tensor.shape[0])
    return {"obs": obs_dict, "actions": actions_tensor, "length": length}


def episode_to_reference(episode, args):
    """Convert a rollout episode into position/velocity/acceleration arrays."""
    pr = episode["obs"]["position"].cpu().numpy().astype(np.float32, copy=False)
    vr = episode["obs"]["velocity"].cpu().numpy().astype(np.float32, copy=False)
    actions = episode["actions"].cpu().numpy().astype(np.float32, copy=False)

    if pr.shape[0] == 0:
        raise RuntimeError("Bootstrap rollout produced an empty trajectory.")

    # Approximate plant dynamics used by the controller:
    #   p_ddot ~= (gear * action) / mass
    ar = ((args.gear / args.mass) * actions).astype(np.float32, copy=False)
    return pr, vr, ar


def main():
    args = parse_args()
    if args.num_trajs <= 0:
        raise ValueError("--num-trajs must be positive.")
    if args.num_rollouts_per_spline <= 0:
        raise ValueError("--num-rollouts-per-spline must be positive.")
    if args.action_noise_var < 0.0:
        raise ValueError("--action-noise-var must be non-negative.")
    args.action_noise_std = math.sqrt(args.action_noise_var)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    env = suite.load(
        domain_name="point_mass",
        task_name=args.task,
        task_kwargs={"random": args.seed, "time_limit": args.time_limit_sec},
    )

    dt = float(env.control_timestep())
    episode_groups = []
    group_splines = []
    total_episodes = 0

    for traj_idx in tqdm(range(args.num_trajs), desc="Collecting spline groups", unit="group"):
        _, pr, vr, ar = generate_spline_noise_traj_2d(
            dt=dt,
            horizon_sec=args.horizon_sec,
            n_anchors=args.n_anchors,
            velocity_scale=args.velocity_scale,
            seed=args.seed + traj_idx,
        )
        scale_to_accel_limit_2d(vr, ar, amax=args.amax)

        # Pass 1: track the sampled spline once (without control noise) and use
        # that rollout as the feasible reference for all saved episodes.
        bootstrap_noise_std = args.action_noise_std
        args.action_noise_std = 0.0
        bootstrap_episode = collect_one_episode(env, rng, args, pr, vr, ar)
        args.action_noise_std = bootstrap_noise_std

        pr_ref, vr_ref, ar_ref = episode_to_reference(bootstrap_episode, args)

        group = []
        for _ in range(args.num_rollouts_per_spline):
            # Pass 2: collect rollouts that track the rollout-derived trajectory.
            group.append(collect_one_episode(env, rng, args, pr_ref, vr_ref, ar_ref))
            total_episodes += 1
        episode_groups.append(group)
        # Save the feasible reference directly in episode format (obs/actions/length)
        # so downstream datasets can consume it without re-deriving actions/states.
        group_splines.append(
            {
                "obs": bootstrap_episode["obs"],
                "actions": bootstrap_episode["actions"],
                "length": bootstrap_episode["length"],
            }
        )

    payload = {
        "episode_groups": episode_groups,
        "group_splines": group_splines,
        "num_episodes": total_episodes,
        "num_similar_trajectories": args.num_rollouts_per_spline,
    }
    torch.save(payload, out_path)
    print(
        f"Saved {len(episode_groups)} groups "
        f"x {args.num_rollouts_per_spline} rollouts to {out_path}"
    )


if __name__ == "__main__":
    main()
