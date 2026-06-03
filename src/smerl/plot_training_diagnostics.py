"""Plot SMERL training diagnostics to diagnose the gate/diversity oscillation.

Top:    skill separation (std of per-skill episode length) and mean success.
Bottom: critic mean-Q and critic loss — the off-policy value non-stationarity
        driven by the frozen, gate-dependent rewards in the replay buffer.
Shaded by per-window gate-fire rate (gated_inc). If the Q swings and critic-loss
spikes line up with separation swings + gate flips, the stale-reward story holds.

    python -m src.smerl.plot_training_diagnostics --run runs/smerl_lowalpha_ckpt
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    args = ap.parse_args()

    d = json.load(open(os.path.join("src/smerl", args.run, "summary.json")))
    H = d["history"]
    step = np.array([h["step"] for h in H])
    succ = np.array([h["mean_success_across_skills"] for h in H])
    sep = np.array([np.std([s["mean_length"] for s in h["per_skill"]]) for h in H])
    gated = np.array([h.get("gated_inc", np.nan) for h in H])
    mean_q = np.array([h.get("mean_q", np.nan) for h in H], dtype=float)
    closs = np.array([h.get("critic_loss", np.nan) for h in H], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- top: separation vs success ---
    ax1.plot(step, sep, "o-", color="tab:purple", label="skill separation (std of length)")
    ax1.set_ylabel("separation (std episode length)", color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax1b = ax1.twinx()
    ax1b.plot(step, succ, "s--", color="tab:green", label="mean success")
    ax1b.set_ylabel("mean success", color="tab:green")
    ax1b.tick_params(axis="y", labelcolor="tab:green")
    ax1b.set_ylim(-0.05, 1.05)
    ax1.set_title(f"{args.run} — separation vs success (they move in opposition)")
    ax1.grid(alpha=0.3)

    # --- bottom: critic value non-stationarity ---
    ax2.plot(step, mean_q, "o-", color="tab:red", label="mean Q")
    ax2.set_ylabel("mean Q", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2b = ax2.twinx()
    ax2b.plot(step, closs, "^--", color="tab:blue", label="critic loss")
    ax2b.set_ylabel("critic loss", color="tab:blue")
    ax2b.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_xlabel("env step")
    ax2.set_title("critic non-stationarity (Q swings as the gated/ungated reward mix turns over)")
    ax2.grid(alpha=0.3)

    # shade by gate-fire rate on both panels
    for ax in (ax1, ax2):
        for i in range(len(step) - 1):
            if not np.isnan(gated[i]):
                ax.axvspan(step[i], step[i + 1], color="orange",
                           alpha=0.12 * gated[i], lw=0)

    plt.tight_layout()
    out = os.path.join("src/smerl", args.run, "training_diagnostics.png")
    plt.savefig(out, dpi=130)
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
