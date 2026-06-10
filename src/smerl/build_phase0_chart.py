"""Aggregate the Phase-0 batch-8 multi-seed sweep and render a grouped bar chart.

Configs: positional encoding {wpe, rope} x switch_share {0.2,0.3,0.4,0.5} x seed {0,1,2}.
  seed 0  -> eval_bs8_share{sh}.log (wpe) / eval_bs8_rope_share{sh}.log (rope)
  seed 1,2-> eval_bc_bs8_{wpe|rope}_share{sh}_seed{sd}.log
Each log's ".pt" row holds: success, succ|failed, %failed, switches.

    python -m src.smerl.build_phase0_chart
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = "src/smerl/runs/smerl_lowalpha_ckpt"
SHARES = [20, 30, 40, 50]
POS = ["wpe", "rope"]
SEEDS = [0, 1, 2]


def log_path(pos, sh, sd):
    # uniform seedN-style log takes priority (re-evals land here); fall back to the
    # original seed-0 naming.
    cand = os.path.join(RUN, f"eval_bc_bs8_{pos}_share{sh}_seed{sd}.log")
    if os.path.exists(cand):
        return cand
    if sd == 0:
        return os.path.join(RUN, f"eval_bs8_{'rope_' if pos == 'rope' else ''}share{sh}.log")
    return cand


def parse(path):
    """Return (success, succ_failed, pct_failed, switches) from the .pt row, or None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if ".pt" in line:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line.split(".pt")[1])
                if len(nums) >= 4:
                    return tuple(float(x) for x in nums[:4])
    return None


def main():
    # data[pos][sh] = list of (success, succ|failed, %failed, switches) over seeds
    data = {p: {sh: [] for sh in SHARES} for p in POS}
    missing = []
    for p in POS:
        for sh in SHARES:
            for sd in SEEDS:
                r = parse(log_path(p, sh, sd))
                if r is None:
                    missing.append(f"{p}/share{sh}/seed{sd}")
                else:
                    data[p][sh].append(r)
    if missing:
        print(f"[warn] {len(missing)} missing eval logs: {', '.join(missing)}")

    # ---- text table ----
    print(f"\n{'pos':>5} {'share':>6} {'n':>2}  {'success(mean±range)':>22}  "
          f"{'succ|failed':>20}  {'switches':>9}")
    summary = {p: {} for p in POS}
    for p in POS:
        for sh in SHARES:
            rows = np.array(data[p][sh]) if data[p][sh] else np.zeros((0, 4))
            if len(rows) == 0:
                continue
            succ, sf, _, sw = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
            summary[p][sh] = (succ.mean(), succ.min(), succ.max(),
                              sf.mean(), sf.min(), sf.max(), sw.mean())
            print(f"{p:>5} {sh/100:>6.2f} {len(rows):>2}  "
                  f"{succ.mean():>6.3f} [{succ.min():.3f},{succ.max():.3f}]  "
                  f"{sf.mean():>6.3f} [{sf.min():.3f},{sf.max():.3f}]  {sw.mean():>9.2f}")

    # ---- grouped bar chart: success + succ|failed, wpe vs rope, x=share ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = np.arange(len(SHARES))
    w = 0.36
    colors = {"wpe": "#3b78c2", "rope": "#d9822b"}
    for ax, (mi, title) in zip(axes, [(0, "overall success"),
                                       (3, "success | failure injected")]):
        for k, p in enumerate(POS):
            means, los, his = [], [], []
            for sh in SHARES:
                s = summary[p].get(sh)
                if s is None:
                    means.append(0); los.append(0); his.append(0); continue
                m, lo, hi = s[mi], s[mi + 1], s[mi + 2]
                means.append(m); los.append(m - lo); his.append(hi - m)
            xb = x + (k - 0.5) * w
            ax.bar(xb, means, w, label=p, color=colors[p],
                   yerr=[los, his], capsize=4, edgecolor="k", linewidth=0.5)
            for xi, m in zip(xb, means):
                if m > 0:
                    ax.text(xi, m + 0.012, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f"{sh/100:.1f}" for sh in SHARES])
        ax.set_xlabel("switch_share")
        ax.set_title(title); ax.set_ylim(0, 1.08); ax.grid(axis="y", alpha=0.3)
        ax.legend(title="pos-enc", loc="lower center")
    axes[0].set_ylabel("rate (mean over 3 seeds; bars=min/max)")
    fig.suptitle("Phase-0 batch-8 sweep: switch_share x positional encoding (skills 1,2, "
                 "200 eps)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(RUN, "phase0_bs8_seedsweep_bar.png")
    plt.savefig(out, dpi=130)
    print(f"\n[viz] saved {out}")
    return out


if __name__ == "__main__":
    main()
