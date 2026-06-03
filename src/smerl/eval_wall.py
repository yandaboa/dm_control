"""Physical-obstacle adaptation eval: NO injected failures — a wall blocks the
skill the policy starts with, so its state stops progressing and the exposed value
stagnates, which should trigger a switch to the unobstructed skill.

Fixed (un-noised) start. For each trial a random wall is placed on the start
skill's path (verified to block it while sparing the other skill). The policy
(trained only on synthetic failures) is run in a reset-on-switch BAMDP with theta=0
(no failures) + the normalized value. We report whether it adapts to the goal.

    python -m src.smerl.eval_wall --model bc_logic_dec.pt --n-trials 60
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env
from src.smerl.train_task_value import load_value_net
from src.smerl.bamdp_env import BAMDPConfig, SyntheticFailureBAMDP
from src.smerl.collect_trajectories import wrap_value_norm
from src.smerl.eval_bc_multimodal import load_bc
from src.smerl.wall_env import WallPoint2DGoalEnv, rollout_path, valid_walls
from src.smerl.visualize_bamdp_switch import rollout_record


@torch.no_grad()
def model_start_skill(model, start, device):
    sid, D = model.id_of["state"], model.max_dim
    sv = np.zeros(D, np.float32); sv[:len(start)] = start
    tid = torch.tensor([[sid]], device=device)
    tval = torch.as_tensor(sv[None, None], device=device)
    h = model.backbone(model.embed(tid, tval),
                       torch.ones_like(tid, dtype=torch.float32))
    return int(model.head_logits("skill", h[0, -1]).argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--model", type=str, default="bc_logic_dec.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--skills", type=str, default="1,2")
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--n-plot", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--wall-radius", type=lambda s: [float(x) for x in s.split(",")],
                    default=[0.05, 0.06, 0.07, 0.08],
                    help="comma-separated candidate wall radii (small -> large)")
    ap.add_argument("--force-start", type=int, default=None,
                    help="force the policy to begin committed to this skill (the wall "
                         "is placed on its path). Default: the model's natural start skill.")
    ap.add_argument("--out", type=str, default="wall_adaptation.png")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = os.path.join("src/smerl", args.run)
    agent, cfg, nominal_start, _ = load_agent(
        os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    goal = build_env(cfg).goal
    nominal_start = np.asarray(nominal_start, np.float32)
    n_skills = cfg["n_skills"]
    sr = cfg.get("success_radius", 0.25)
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    allowed = [int(s) for s in args.skills.split(",")]
    value_net, _, _ = wrap_value_norm(value_net, allowed, agent, cfg, nominal_start,
                                      0.0, device, n_skills)
    model = load_bc(os.path.join(run_dir, args.model), device)

    def env_fn(wall):
        return WallPoint2DGoalEnv(start=tuple(nominal_start), goal=tuple(goal),
                                  wall=wall, success_radius=sr,
                                  max_episode_steps=args.max_steps)

    if args.force_start is not None:
        z_block = args.force_start                  # the policy is forced onto this skill
    else:
        z_block = model_start_skill(model, nominal_start, device)
    z_spare = [s for s in allowed if s != z_block][0]
    print(f"[wall] policy starts with skill {z_block} "
          f"({'forced' if args.force_start is not None else 'natural'}); wall sticks it, "
          f"skill {z_spare} is the escape  ->  showing {z_block}->{z_spare} switches")

    rng = np.random.default_rng(args.seed)
    walls = valid_walls(agent, env_fn, z_block, z_spare, rng,
                        radii=tuple(args.wall_radius), cap=args.n_trials)
    print(f"[wall] found {len(walls)} valid walls (stick {z_block}, spare {z_spare})")
    n_succ = n_switch = n_switch_to_spare = n_t = 0
    picks = []
    for wall in walls:
        n_t += 1
        cfg_b = BAMDPConfig(n_skills=n_skills,
                            skill_lengths=np.full(n_skills, float(args.max_steps)),
                            n_sub_episodes=10**9, schedule="budget",
                            continue_on_failure=True, expose_value=True,
                            expose_failing=True, certain_skill=True,
                            reset_on_switch=True)
        bamdp = SyntheticFailureBAMDP(lambda r, w=wall: env_fn(w), agent.disc, cfg_b,
                                      np.random.default_rng(1), device,
                                      value_net=value_net)
        bamdp.reset_meta(theta=np.zeros(n_skills))   # NO injected failures
        ep = rollout_record(model, bamdp, device, force_first=args.force_start)
        ep["wall"] = wall
        n_succ += int(ep["success"])
        if ep["switches"]:
            n_switch += 1
            if int(ep["skill"][-1]) == z_spare:
                n_switch_to_spare += 1
        if ep["switches"] and ep["success"] and len(picks) < args.n_plot:
            picks.append(ep)

    print(f"[wall] {n_t} trials with a valid wall (no injected failures):")
    print(f"  reached goal:            {n_succ}/{n_t} = {n_succ/max(n_t,1):.3f}")
    print(f"  switched (adapted):      {n_switch}/{n_t} = {n_switch/max(n_t,1):.3f}")
    print(f"  switched to escape skill:{n_switch_to_spare}/{max(n_switch,1)}")

    if picks:
        cmap = plt.get_cmap("tab10")
        N = len(picks)
        fig, axes = plt.subplots(2, N, figsize=(3.6 * N, 6.4), squeeze=False,
                                 gridspec_kw={"height_ratios": [3, 2]})
        for c, ep in enumerate(picks):
            xy, sk, val, tel = ep["xy"], ep["skill"], ep["value"], ep["teleports"]
            sw = ep["switches"][0]; wcx, wcy, wrho = ep["wall"]
            ax = axes[0, c]
            ax.add_patch(plt.Circle((wcx, wcy), wrho, color="0.3", alpha=0.7,
                                    zorder=1))                              # wall
            for t in range(len(xy) - 1):
                if tel[t + 1]:
                    continue
                ax.plot(xy[t:t + 2, 0], xy[t:t + 2, 1], color=cmap(sk[t]), lw=2, zorder=2)
            ax.scatter(*xy[0], marker="*", c="k", s=200, zorder=5)
            ax.scatter(goal[0], goal[1], marker="X", c="k", s=150, zorder=5)
            ax.add_patch(plt.Circle(goal, sr, color="k", fill=False, ls="--", alpha=0.4))
            ax.scatter(*xy[sw - 1], marker="s", facecolors="none", edgecolors="r",
                       s=120, linewidths=2, zorder=6)
            ax.scatter(*xy[sw], marker="*", c="lime", s=180, edgecolors="k", zorder=6)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal"); ax.grid(alpha=.3)
            ax.set_title(f"stuck {z_block} → {int(sk[-1])} (switch@{sw})", fontsize=10)
            axb = axes[1, c]
            axb.plot(val, color="0.5", lw=1)
            axb.scatter(range(len(val)), val, c=[cmap(z) for z in sk], s=16)
            axb.axvline(sw, color="k", lw=1.5, label="switch")
            axb.set_ylim(-0.02, 1.02); axb.set_xlabel("timestep")
            if c == 0:
                axb.set_ylabel("value (normalized)"); axb.legend(fontsize=8)
            axb.grid(alpha=.3)
        fig.suptitle(f"WALL adaptation (no injected failures) — {args.model}  "
                     f"[{z_block}→{z_spare}]\n"
                     f"grey=wall; value stalls while stuck, then switch+retry to goal",
                     fontsize=12)
        plt.tight_layout()
        out = os.path.join(run_dir, args.out)
        plt.savefig(out, dpi=130)
        print(f"[viz] saved {out}")


if __name__ == "__main__":
    main()
