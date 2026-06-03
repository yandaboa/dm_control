"""Value graphs from the reset-on-switch intervention histories.

For a few stored DAgger episodes (split by which skill is bad, z0), plot the
EXPOSED value over time (what the model sees: corrupted after failure) against the
TRUE value V(s_t, active_skill) recomputed from the value net (what it would be
with no failure). The gap between them after the failure is the "should switch"
signal — this shows how legible it actually is, and whether it differs between the
fast skill (0) and the slow skill (4) being the bad one.

    python -m src.smerl.visualize_value_intervention
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.smerl.skill_decode import load_agent
from src.smerl.train_task_value import load_value_net
from src.smerl.trajectory_store import load_manifest, load_episode


@torch.no_grad()
def true_value(value_net, states, skills, device):
    """V(s_t, active_skill_t) recomputed (uncorrupted) from the value net."""
    S = torch.as_tensor(np.asarray(states, np.float32), device=device)
    z = torch.as_tensor(np.asarray(skills, np.int64), device=device)
    return value_net(S, z).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--store", type=str, default="trajectories_dagger2")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--per-case", type=int, default=3)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = os.path.join("src/smerl", args.run)
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    store = os.path.join(run_dir, args.store)
    man = load_manifest(store)

    # split by which skill is bad (z0 = the attempted/failed skill)
    cases = {0: [], 4: []}
    for rec in man["episodes"]:
        if rec.get("intervened") and rec["z"] in cases:
            cases[rec["z"]].append(rec)
    cmap = plt.get_cmap("tab10")

    ncol = args.per_case
    fig, axes = plt.subplots(2, ncol, figsize=(4 * ncol, 6.4), squeeze=False)
    for row, z0 in enumerate([0, 4]):
        recs = cases[z0][:ncol]
        for c, rec in enumerate(recs):
            ep = load_episode(store, rec)
            S = ep["states"][:-1]; sk = ep["skills"]; V = ep["values"][:-1]
            tv = true_value(value_net, S, sk, device)
            fs = rec["fail_step"]; isw = rec["intervene_step"]; zg = rec["z_good"]
            ax = axes[row, c]
            # exposed value, colored by executed skill
            ax.scatter(range(len(V)), V, c=[cmap(int(z)) for z in sk], s=16, zorder=3,
                       label="exposed (model sees)")
            ax.plot(V, color="0.6", lw=0.8, zorder=2)
            ax.plot(tv, color="k", ls="--", lw=1.2, zorder=4,
                    label="true V(s, skill)")
            if fs:
                ax.axvline(fs - 1, color="r", ls=":", lw=1.5, label="failure")
            if isw:
                ax.axvline(isw - 1, color="k", ls="-", lw=1.2, label="switch")
            ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3)
            ax.set_title(f"bad z0={z0} → z_good={zg}  (fail@{fs})", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"value  (z0={z0} {'fast' if z0==0 else 'SLOW'})")
                ax.legend(fontsize=7, loc="upper left")
            if row == 1:
                ax.set_xlabel("timestep")
    fig.suptitle("Intervention value graphs — exposed (corrupted) vs true V(s,skill)\n"
                 "the gap after 'failure' is the switch signal; note how flat it is "
                 "when the SLOW skill (4) is bad", fontsize=12)
    plt.tight_layout()
    out = os.path.join(run_dir, "value_intervention.png")
    plt.savefig(out, dpi=130)
    print(f"[viz] saved {out}")
    # quantify the gap: mean (true - exposed) over the stalled window, per case
    for z0 in [0, 4]:
        gaps = []
        for rec in cases[z0][:80]:
            ep = load_episode(store, rec); S = ep["states"][:-1]; sk = ep["skills"]
            V = ep["values"][:-1]; tv = true_value(value_net, S, sk, device)
            fs = rec["fail_step"]; isw = rec["intervene_step"]
            if fs and isw and isw > fs:
                gaps.append(float(np.mean(tv[fs-1:isw-1] - V[fs-1:isw-1])))
        print(f"  bad z0={z0}: mean (true-exposed) over stall window = "
              f"{np.mean(gaps):.3f}  (n={len(gaps)})")


if __name__ == "__main__":
    main()
