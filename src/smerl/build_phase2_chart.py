"""Phase-2 (3 skills 0,1,2) cross-episode latching across on-policy rounds r1..r3.
Shows the "rounds scale with #skills" thesis: success & P(first==bad) by sub-episode.

    python -m src.smerl.build_phase2_chart
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "src/smerl/runs/smerl_lowalpha_ckpt"
ROUNDS = [
    ("round 1 (1 on-policy round)", "eval_phase2_r1_multiep.log", "#c0392b"),
    ("round 2", "eval_phase2_r2_multiep.log", "#2e86c1"),
    ("round 3", "eval_phase2_r3_multiep.log", "#1e8449"),
]


def parse_curves(path):
    if not os.path.exists(path):
        return None, None
    rows = []
    with open(path) as f:
        for line in f:
            if re.search(r"#\d+:", line):
                rows.append([float(x) for x in re.findall(r"#\d+:([0-9.]+)", line)])
    return (rows[0], rows[1]) if len(rows) >= 2 else (None, None)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, fn, color in ROUNDS:
        succ, pbad = parse_curves(os.path.join(RUN, fn))
        if succ is None:
            print(f"[skip] {label}: no data ({fn})")
            continue
        x = list(range(len(succ)))
        axes[0].plot(x, succ, color=color, marker="o", lw=2, label=label)
        axes[1].plot(x, pbad, color=color, marker="o", lw=2, label=label)
        print(f"{label:>26}  success={succ}  P(bad)={pbad}")
    axes[1].axhline(1/3, color="0.5", ls=":", lw=1, label="chance (1/3)")
    axes[0].set_title("success rate by sub-episode  (latching → stay high)")
    axes[0].set_ylabel("success rate"); axes[0].set_ylim(0, 1.05)
    axes[1].set_title("P(first skill == bad)  (avoidance → fall below 1/3)")
    axes[1].set_ylabel("P(first == bad)"); axes[1].set_ylim(0, 0.6)
    for ax in axes:
        ax.set_xlabel("sub-episode index (shared θ, persistent context)")
        ax.set_xticks([0, 1, 2, 3]); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("Phase 2 — cross-episode latching, 3 skills (0,1,2): more on-policy "
                 "rounds → better coverage", fontsize=12)
    plt.tight_layout()
    out = os.path.join(RUN, "phase2_latching_rounds.png")
    plt.savefig(out, dpi=130)
    print(f"\n[viz] saved {out}")
    return out


if __name__ == "__main__":
    main()
