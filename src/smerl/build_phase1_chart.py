"""Phase-1 cross-episode latching: success & P(first==bad) by sub-episode index,
across round-1 (no latching) vs round-2 (long-supervision fix). Parses the
eval_adapt_multiep logs.

    python -m src.smerl.build_phase1_chart
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "src/smerl/runs/smerl_lowalpha_ckpt"

# (label, logfile, color, linestyle)
SERIES = [
    ("round-1 rope+0.2", "eval_phase1_rope_multiep.log", "#c0392b", "--"),
    ("round-1 wpe+0.4",  "eval_phase1_wpe_multiep.log",  "#e59866", ":"),
    ("round-2 share0.2", "eval_phase1_r2_multiep.log",   "#2e86c1", "-"),
    ("round-2 share0.4", "eval_phase1_r2_s40_multiep.log","#1e8449", "-"),
]


def parse_curves(path):
    """Return (success_list, pbad_list) from the two '#i:val' lines, or (None,None)."""
    if not os.path.exists(path):
        return None, None
    rows = []
    with open(path) as f:
        for line in f:
            if re.search(r"#\d+:", line):
                rows.append([float(x) for x in re.findall(r"#\d+:([0-9.]+)", line)])
    if len(rows) < 2:
        return None, None
    return rows[0], rows[1]  # success, P(first==bad)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, fn, color, ls in SERIES:
        succ, pbad = parse_curves(os.path.join(RUN, fn))
        if succ is None:
            print(f"[skip] {label}: no data ({fn})")
            continue
        x = list(range(len(succ)))
        axes[0].plot(x, succ, color=color, ls=ls, marker="o", lw=2, label=label)
        axes[1].plot(x, pbad, color=color, ls=ls, marker="o", lw=2, label=label)
        print(f"{label:>18}  success={succ}  P(bad)={pbad}")

    axes[0].set_title("success rate by sub-episode  (latching → should RISE / stay high)")
    axes[0].set_ylabel("success rate"); axes[0].set_ylim(0, 1.05)
    axes[1].set_title("P(first skill == bad)  (avoidance → should FALL)")
    axes[1].set_ylabel("P(first == bad)"); axes[1].set_ylim(0, 1.0)
    for ax in axes:
        ax.set_xlabel("sub-episode index (shared θ, persistent context)")
        ax.set_xticks([0, 1, 2, 3]); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("Phase 1 — cross-episode latching (skills 1,2): round-1 fails, "
                 "round-2 long-supervision fix", fontsize=12)
    plt.tight_layout()
    out = os.path.join(RUN, "phase1_latching_curves.png")
    plt.savefig(out, dpi=130)
    print(f"\n[viz] saved {out}")
    return out


if __name__ == "__main__":
    main()
