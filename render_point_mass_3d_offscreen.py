#!/usr/bin/env python3
"""Headless offscreen renderer for dm_control point_mass tasks (2D and 3D)."""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

# Set before importing dm_control so the renderer backend is chosen correctly.
os.environ.setdefault("MUJOCO_GL", "egl")

from dm_control import suite  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render point_mass task offscreen and save PNG frames."
    )
    parser.add_argument(
        "--task",
        default="easy",
        choices=["easy", "hard", "easy_3d", "hard_3d"],
        help="point_mass task name from dm_control.suite.point_mass",
    )
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-id", default="fixed")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for PNG frames. "
        "Default: renders/point_mass_<task>",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir or f"renders/point_mass_{args.task}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    env = suite.load(domain_name="point_mass", task_name=args.task)
    action_spec = env.action_spec()
    time_step = env.reset()

    for i in range(args.frames):
        action = rng.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum,
            size=action_spec.shape,
        ).astype(action_spec.dtype)
        time_step = env.step(action)

        pixels = env.physics.render(
            height=args.height,
            width=args.width,
            camera_id=args.camera_id,
        )
        Image.fromarray(pixels).save(out_dir / f"frame_{i:05d}.png")

        if time_step.last():
            time_step = env.reset()

    print(f"Saved {args.frames} frames to {out_dir}")


if __name__ == "__main__":
    main()
