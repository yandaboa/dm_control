"""Visualize DAgger TRAINING episodes — what a switch actually looks like in the
supervised data.

For a few stored intervention episodes (one per column): the xy path colored by
the EXECUTED skill (z0 then z_good), with the failure onset and the execution
switch marked; below it the exposed value over time, dots colored by the
DECOUPLED skill TARGET (so you see where the label flips to z_good vs where the
execution actually switches). Tests the "states are too similar after switching"
hypothesis: if the path doesn't visibly change at the switch, the only thing
distinguishing pre/post-switch is the value token.

    python -m src.smerl.visualize_dagger_training --n 4
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.smerl.trajectory_store import load_manifest, load_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--store", type=str, default="trajectories_dagger")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--obs-delay", type=int, default=2)
    args = ap.parse_args()

    store = os.path.join("src/smerl", args.run, args.store)
    man = load_manifest(store)
    goal = np.asarray(man["meta"]["goal"])[:2]
    n_skills = int(man["meta"]["n_skills"])

    # pick varied intervention episodes: distinct (z0, z_good) and spread fail_step
    cand = [r for r in man["episodes"]
            if r.get("intervened") and r["success"] and r["fail_step"]]
    cand.sort(key=lambda r: r["fail_step"])
    picks, seen = [], set()
    for r in cand:                                   # spread of fail timing + pairs
        key = (r["z"], r["z_good"])
        if key not in seen or len(picks) < args.n:
            picks.append(r); seen.add(key)
        if len(picks) >= args.n:
            break
    idx = np.linspace(0, len(cand) - 1, args.n).astype(int)
    picks = [cand[i] for i in idx]                   # evenly spread by fail_step

    cmap = plt.get_cmap("tab10")
    N = len(picks)
    fig, axes = plt.subplots(2, N, figsize=(3.6 * N, 6.4), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 2]})
    for c, rec in enumerate(picks):
        ep = load_episode(store, rec)
        S = ep["states"][:-1]; xy = S[:, :2]
        sk = ep["skills"]; st = ep.get("skill_target"); V = ep["values"][:-1]
        fs = rec["fail_step"]; isw = rec["intervene_step"]
        z0, zg = int(rec["z"]), int(rec["z_good"])
        tflip = (fs - 1 + args.obs_delay) if fs else None

        ax = axes[0, c]
        for t in range(len(xy) - 1):
            ax.plot(xy[t:t + 2, 0], xy[t:t + 2, 1], color=cmap(sk[t]), lw=2, zorder=2)
        ax.scatter(*xy[0], marker="*", c="k", s=220, zorder=5)
        ax.scatter(goal[0], goal[1], marker="X", c="k", s=160, zorder=5)
        ax.add_patch(plt.Circle(goal, man["meta"].get("success_radius", 0.25),
                                color="k", fill=False, ls="--", alpha=0.4))
        if fs and fs - 1 < len(xy):
            ax.scatter(*xy[fs - 1], marker="s", c="r", s=70, zorder=6,
                       label="failure")
        if isw and isw - 1 < len(xy):
            ax.scatter(*xy[isw - 1], marker="o", facecolors="none",
                       edgecolors="k", s=220, linewidths=2, zorder=6,
                       label="exec switch")
        ax.set_xlim(-1.6, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.grid(alpha=.3)
        ax.set_title(f"skill {z0} → {zg}   fail@{fs}, switch@{isw}", fontsize=10)
        if c == 0:
            ax.legend(fontsize=7, loc="lower left")

        axb = axes[1, c]
        axb.plot(V, color="0.5", lw=1, zorder=1)
        if st is not None:
            axb.scatter(range(len(V)), V, c=[cmap(int(z)) for z in st], s=16, zorder=2)
        if fs:
            axb.axvline(fs - 1, color="r", ls=":", lw=1.5, label="failure")
        if tflip is not None:
            axb.axvline(tflip, color="orange", ls="--", lw=1.5, label="target→z_good")
        if isw:
            axb.axvline(isw - 1, color="k", ls="-", lw=1.5, label="exec switch")
        axb.set_ylim(-0.02, 1.02); axb.set_xlabel("timestep")
        if c == 0:
            axb.set_ylabel("exposed value"); axb.legend(fontsize=7, loc="lower right")
        axb.grid(alpha=.3)

    for z in range(n_skills):
        axes[0, 0].plot([], [], color=cmap(z), lw=3, label=f"skill {z}")
    axes[0, 0].legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("DAgger TRAINING episodes — path colored by executed skill "
                 "(z0→z_good); dots below colored by decoupled skill TARGET",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join("src/smerl", args.run, "dagger_training_episodes.png")
    plt.savefig(out, dpi=130)
    print(f"[viz] saved {out}  ({N} episodes)")


if __name__ == "__main__":
    main()
