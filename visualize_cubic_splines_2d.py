#!/usr/bin/env python3
"""Generate and visualize random 2D cubic splines to a local image."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from dm_control.utils.cubic_spline import generate_spline_noise_traj_2d
from dm_control.utils.cubic_spline import scale_to_accel_limit_2d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a few random cubic splines and plot them in 2D."
    )
    parser.add_argument("--num-splines", type=int, default=4)
    parser.add_argument("--horizon-sec", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--n-anchors", type=int, default=10)
    parser.add_argument("--velocity-scale", type=float, default=0.30)
    parser.add_argument("--amax", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-path",
        default="renders/cubic_splines_2d.png",
        help="Path to output image file.",
    )
    return parser.parse_args()


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    low = np.array([-0.28, -0.28], dtype=float)
    high = np.array([0.28, 0.28], dtype=float)

    # Draw workspace bounds.
    rect_xy = low
    rect_wh = high - low
    ax.add_patch(
        plt.Rectangle(
            rect_xy,
            rect_wh[0],
            rect_wh[1],
            fill=False,
            linewidth=1.5,
            linestyle="--",
        )
    )

    for i in range(args.num_splines):
        t, pr, vr, ar = generate_spline_noise_traj_2d(
            dt=args.dt,
            horizon_sec=args.horizon_sec,
            n_anchors=args.n_anchors,
            velocity_scale=args.velocity_scale,
            seed=args.seed + i,
        )
        scale_to_accel_limit_2d(vr, ar, amax=args.amax)
        del t, vr, ar

        ax.plot(pr[:, 0], pr[:, 1], linewidth=1.8, label=f"spline {i}")
        ax.scatter(pr[0, 0], pr[0, 1], marker="o", s=18)
        ax.scatter(pr[-1, 0], pr[-1, 1], marker="x", s=24)

    ax.set_title("Random 2D Cubic Splines")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", "box")
    ax.set_xlim(low[0] - 0.02, high[0] + 0.02)
    ax.set_ylim(low[1] - 0.02, high[1] + 0.02)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"Saved spline visualization to {out_path}")


if __name__ == "__main__":
    main()
