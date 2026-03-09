#!/usr/bin/env python3
"""Visualize grouped point-mass episodes against their reference splines."""

import argparse
from pathlib import Path

import matplotlib
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot each episode group over its reference spline."
    )
    parser.add_argument(
        "--episodes-path",
        required=True,
        help="Path to .pt file saved by collect_point_mass_spline_trajectories.py",
    )
    parser.add_argument(
        "--out-dir",
        default="renders/point_mass_episode_groups",
        help="Directory for generated group plots.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional max number of groups to render.",
    )
    return parser.parse_args()


def _to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return tensor_or_array


def _reference_positions(group_reference):
    """Extract 2D reference positions from either legacy or episode-style format."""
    # New format: group_splines[k] stores an episode-like dict with obs/actions/length.
    if isinstance(group_reference, dict) and "obs" in group_reference:
        obs = group_reference["obs"]
        if isinstance(obs, dict) and "position" in obs:
            return _to_numpy(obs["position"])
        raise ValueError("Expected group reference obs to contain 'position'.")

    # Legacy format: group_splines[k] stores direct trajectory tensors.
    if isinstance(group_reference, dict) and "position" in group_reference:
        return _to_numpy(group_reference["position"])

    raise ValueError("Unrecognized group reference format in 'group_splines'.")


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.episodes_path)
    episode_groups = payload["episode_groups"]
    group_splines = payload.get("group_splines")

    if group_splines is None:
        raise ValueError(
            "Missing 'group_splines' in payload. "
            "Re-collect data with the updated collector script."
        )
    if len(group_splines) != len(episode_groups):
        raise ValueError("group_splines and episode_groups length mismatch.")

    n_groups = len(episode_groups)
    if args.max_groups is not None:
        n_groups = min(n_groups, args.max_groups)
    
    MAX_NUM_GROUPS = 20

    for group_idx in range(n_groups):
        if group_idx >= MAX_NUM_GROUPS:
            break
        group = episode_groups[group_idx]
        reference = group_splines[group_idx]
        ref_pos = _reference_positions(reference)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(
            ref_pos[:, 0],
            ref_pos[:, 1],
            "k--",
            linewidth=2.0,
            label="reference trajectory",
        )
        ax.scatter(ref_pos[0, 0], ref_pos[0, 1], marker="o", s=35, c="k")
        ax.scatter(ref_pos[-1, 0], ref_pos[-1, 1], marker="x", s=40, c="k")

        for rollout_idx, episode in enumerate(group):
            pos = _to_numpy(episode["obs"]["position"])
            ax.plot(pos[:, 0], pos[:, 1], linewidth=1.4, alpha=0.9, label=f"rollout {rollout_idx}")
            ax.scatter(pos[0, 0], pos[0, 1], marker="o", s=15)

        ax.set_title(f"Group {group_idx}: rollouts vs reference")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", "box")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"group_{group_idx:05d}.png", dpi=180)
        plt.close(fig)

    print(f"Saved {n_groups} group plots to {out_dir}")


if __name__ == "__main__":
    main()
