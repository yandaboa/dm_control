"""Wall-adaptation success rate of the winner (bc_logic_dec, all training tricks),
decomposed to show the value of adaptation.

Which skill the wall blocks is randomized 50/50 and is independent of the policy
(the policy deterministically commits to its start skill z*). So:

  * ~half the episodes the wall blocks the OTHER skill -> the policy's chosen skill
    is unobstructed -> it succeeds immediately, no switch needed ("lucky").
  * ~half the wall blocks z* -> the policy gets stuck and must SWITCH to succeed.

A non-adaptive policy would therefore top out at the luck rate (~0.5); our adaptive
policy recovers the blocked half too, lifting the success rate toward 1.0. The plot
is a cumulative stacked area: immediate (lucky) successes + adaptation successes
(+ any failures), with the 0.5 luck-floor and 1.0 ceiling marked.

    python -m src.smerl.plot_wall_success_rate --model bc_logic_dec.pt --n 100
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
from src.smerl.wall_env import WallPoint2DGoalEnv, valid_walls
from src.smerl.visualize_bamdp_switch import rollout_record


@torch.no_grad()
def model_start_skill(model, start, device):
    sid = model.id_of["state"]; D = model.max_dim
    sv = np.zeros(D, np.float32); sv[:len(start)] = start
    tid = torch.tensor([[sid]], device=device)
    h = model.backbone(model.embed(tid, torch.as_tensor(sv[None, None], device=device)),
                       torch.ones_like(tid, dtype=torch.float32))
    return int(model.head_logits("skill", h[0, -1]).argmax())


def run_episode(model, agent, value_net, env_fn, wall, n_skills, max_steps, device):
    cfg_b = BAMDPConfig(n_skills=n_skills,
                        skill_lengths=np.full(n_skills, float(max_steps)),
                        n_sub_episodes=10**9, schedule="budget",
                        continue_on_failure=True, expose_value=True,
                        expose_failing=True, certain_skill=True, reset_on_switch=True)
    bamdp = SyntheticFailureBAMDP(lambda r, w=wall: env_fn(w), agent.disc, cfg_b,
                                  np.random.default_rng(1), device, value_net=value_net)
    bamdp.reset_meta(theta=np.zeros(n_skills))               # NO injected failures
    ep = rollout_record(model, bamdp, device)                # natural start (commits to z*)
    return bool(ep["success"]), bool(ep["switches"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--model", type=str, default="bc_logic_dec.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--skills", type=str, default="1,2")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--out", type=str, default="wall_success_rate.png")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = os.path.join("src/smerl", args.run)
    agent, cfg, start, _ = load_agent(os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    goal = build_env(cfg).goal
    start = np.asarray(start, np.float32)
    n_skills = cfg["n_skills"]; sr = cfg.get("success_radius", 0.25)
    allowed = [int(s) for s in args.skills.split(",")]
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    value_net, _, _ = wrap_value_norm(value_net, allowed, agent, cfg, start, 0.0,
                                      device, n_skills)
    model = load_bc(os.path.join(run_dir, args.model), device)

    def env_fn(wall):
        return WallPoint2DGoalEnv(start=tuple(start), goal=tuple(goal), wall=wall,
                                  success_radius=sr, max_episode_steps=args.max_steps)

    z_star = model_start_skill(model, start, device)        # the skill the policy commits to
    other = [s for s in allowed if s != z_star][0]
    print(f"[wall-sr] policy commits to skill {z_star}; escape skill {other}")

    rng = np.random.default_rng(args.seed)
    # wall pools: blocking z* (adaptation needed) vs blocking the other skill (lucky)
    pool_block_star = valid_walls(agent, env_fn, z_star, other, rng, cap=120)
    pool_block_other = valid_walls(agent, env_fn, other, z_star, rng, cap=120)
    print(f"[wall-sr] walls blocking z*={z_star}: {len(pool_block_star)}; "
          f"blocking other={other}: {len(pool_block_other)}")

    # exactly half the episodes block z* (must adapt), half block the other (lucky)
    N = args.n; half = N // 2
    specs = ([("adapt", z_star, pool_block_star)] * half
             + [("lucky", other, pool_block_other)] * (N - half))
    order = rng.permutation(N)
    specs = [specs[i] for i in order]

    kinds, succ, sw = [], [], []
    for kind, blk, pool in specs:
        wall = pool[int(rng.integers(len(pool)))]
        s, w = run_episode(model, agent, value_net, env_fn, wall, n_skills,
                           args.max_steps, device)
        kinds.append(kind); succ.append(s); sw.append(w)

    kinds = np.array(kinds); succ = np.array(succ)
    immediate = np.array([k == "lucky" and s for k, s in zip(kinds, succ)])
    adapted = np.array([k == "adapt" and s for k, s in zip(kinds, succ)])
    failed = ~succ
    x = np.arange(1, N + 1)
    cum_imm = np.cumsum(immediate) / x
    cum_adp = np.cumsum(adapted) / x
    cum_fail = np.cumsum(failed) / x

    tot = float(succ.mean())
    f_imm = float(immediate.sum()) / N
    f_adp = float(adapted.sum()) / N
    print(f"[wall-sr] total success {succ.sum()}/{N} = {tot:.3f}  "
          f"(immediate {f_imm:.3f} + adapted {f_adp:.3f}); failures {failed.sum()}")

    # ---- plot: cumulative stacked area ----
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(9.5, 6))
    c_imm, c_adp, c_fail = "#7fc97f", "#1f6f3d", "#d62728"
    ax.fill_between(x, 0, cum_imm, color=c_imm,
                    label=f"immediate success — lucky, no switch ({f_imm:.0%})")
    ax.fill_between(x, cum_imm, cum_imm + cum_adp, color=c_adp,
                    label=f"success via adaptation — switched off the wall ({f_adp:.0%})")
    ax.fill_between(x, cum_imm + cum_adp, cum_imm + cum_adp + cum_fail, color=c_fail,
                    alpha=0.85, label=f"failure ({failed.sum()}/{N})")
    ax.axhline(0.5, color="k", ls="--", lw=1.3)
    ax.text(N * 0.985, 0.5 + 0.012, "luck alone ≈ 0.5  (pick the unobstructed skill)",
            ha="right", va="bottom", fontsize=10.5)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.annotate(f"with adaptation → {tot:.2f}", xy=(N, tot), xytext=(N * 0.62, 0.78),
                fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.set_xlim(1, N); ax.set_ylim(0, 1.03)
    ax.set_xlabel("episode"); ax.set_ylabel("cumulative success rate")
    ax.set_title(f"Wall-adaptation success rate — {args.model}\n"
                 f"random wall blocks either skill; adaptation recovers the blocked half",
                 fontsize=13)
    ax.legend(loc="lower center", framealpha=0.95, fontsize=10.5)
    ax.grid(alpha=.25)
    plt.tight_layout()
    out = os.path.join(run_dir, args.out)
    plt.savefig(out, dpi=140)
    print(f"[viz] saved {out}")


if __name__ == "__main__":
    main()
