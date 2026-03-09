#!/usr/bin/env python3
"""Visualize inferred point-mass rollouts against their context trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import torch


def _torch_load_payload(path: str | Path) -> dict[str, Any]:
    """Load a torch-saved payload with PyTorch>=2.6 compatibility."""
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        message = str(exc)
        if "Weights only load failed" not in message:
            raise
        # In PyTorch 2.6+, torch.load defaults to weights_only=True.
        # Our rollout payload contains numpy objects, so reload as full pickle.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload from {path}, got {type(payload)}.")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot inferred rollouts vs selected context episodes."
    )
    parser.add_argument(
        "--rollouts-path",
        required=True,
        help="Path to .pt file saved by infer_point_mass_supervised.py.",
    )
    parser.add_argument(
        "--out-dir",
        default="renders/point_mass_infer_rollouts",
        help="Directory for generated rollout plots.",
    )
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        help="Optional max number of rollouts to render.",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        default=False,
        help="Overlay group_splines reference when available from context_path payload.",
    )
    return parser.parse_args()


def _to_numpy(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _extract_pos_from_obs(obs: Any):
    """Extract XY trajectory from observation payload."""
    if isinstance(obs, dict):
        if "position" not in obs:
            raise ValueError("Expected observation dict to include 'position'.")
        return _to_numpy(obs["position"])
    obs_arr = _to_numpy(obs)
    if obs_arr.ndim != 2 or obs_arr.shape[1] < 2:
        raise ValueError(f"Expected rollout obs shape [T, >=2], got {getattr(obs_arr, 'shape', None)}")
    return obs_arr[:, :2]


def _try_load_reference(payload: dict[str, Any], rollout: dict[str, Any]):
    context_path = payload.get("context_path")
    group_idx = rollout.get("group_index")
    if context_path is None or group_idx is None:
        return None
    context_file = Path(str(context_path))
    if not context_file.exists():
        return None
    context_payload = _torch_load_payload(context_file)
    group_splines = context_payload.get("group_splines")
    if not isinstance(group_splines, list):
        return None
    if int(group_idx) < 0 or int(group_idx) >= len(group_splines):
        return None
    group_ref = group_splines[int(group_idx)]
    if isinstance(group_ref, dict) and "obs" in group_ref:
        obs = group_ref["obs"]
        if isinstance(obs, dict) and "position" in obs:
            return _to_numpy(obs["position"])
    if isinstance(group_ref, dict) and "position" in group_ref:
        return _to_numpy(group_ref["position"])
    return None


def main() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _torch_load_payload(args.rollouts_path)
    rollouts = payload.get("rollouts")
    if not isinstance(rollouts, list):
        raise ValueError("Expected payload key 'rollouts' to be a list.")

    n_rollouts = len(rollouts)
    if args.max_rollouts is not None:
        n_rollouts = min(n_rollouts, args.max_rollouts)

    for rollout_idx in range(n_rollouts):
        rollout = rollouts[rollout_idx]
        context_episodes = rollout.get("context_episodes", [])
        if not isinstance(context_episodes, list):
            raise ValueError(f"rollout[{rollout_idx}] has invalid 'context_episodes' format.")

        rollout_pos = _extract_pos_from_obs(rollout["obs"])
        ref_pos = _try_load_reference(payload, rollout) if args.include_reference else None

        fig, ax = plt.subplots(figsize=(7, 7))

        if ref_pos is not None:
            ax.plot(
                ref_pos[:, 0],
                ref_pos[:, 1],
                "k--",
                linewidth=2.0,
                label="reference trajectory",
            )
            ax.scatter(ref_pos[0, 0], ref_pos[0, 1], marker="o", s=35, c="k")
            ax.scatter(ref_pos[-1, 0], ref_pos[-1, 1], marker="x", s=40, c="k")

        for ctx_idx, episode in enumerate(context_episodes):
            ctx_pos = _extract_pos_from_obs(episode["obs"])
            context_label = episode.get("context_index", ctx_idx)
            ax.plot(
                ctx_pos[:, 0],
                ctx_pos[:, 1],
                linewidth=1.2,
                alpha=0.8,
                label=f"context {context_label}",
            )
            ax.scatter(ctx_pos[0, 0], ctx_pos[0, 1], marker="o", s=15)

        ax.plot(
            rollout_pos[:, 0],
            rollout_pos[:, 1],
            color="tab:red",
            linewidth=2.0,
            alpha=0.95,
            label="inferred rollout",
        )
        ax.scatter(rollout_pos[0, 0], rollout_pos[0, 1], marker="o", s=28, color="tab:red")
        ax.scatter(rollout_pos[-1, 0], rollout_pos[-1, 1], marker="x", s=36, color="tab:red")

        group_idx = rollout.get("group_index", -1)
        total_reward = float(rollout.get("total_reward", 0.0))
        length = int(rollout.get("length", rollout_pos.shape[0]))
        ax.set_title(
            f"Rollout {rollout_idx}: group={group_idx}, length={length}, return={total_reward:.3f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", "box")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"rollout_{rollout_idx:05d}.png", dpi=180)
        plt.close(fig)

    print(f"Saved {n_rollouts} rollout plots to {out_dir}")


if __name__ == "__main__":
    main()
