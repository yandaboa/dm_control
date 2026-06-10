"""Closed-loop multi-episode adaptation eval (matches collect_demos dynamics).

One bad skill per meta-episode (shared theta); sub-episodes chain under one
persistent transformer context. ``--reset-on-switch`` enables the env-side
terminate-on-switch dynamics (match it to the model's collection dynamics). The model drives on-policy as a
full policy (its own skill AND action heads). We measure two behaviors:

  * switch-after-failure: within a sub-episode, does it abandon a stalled skill?
  * latch-onto-success: across sub-episodes, does it stop picking the bad skill
    (first-step skill) once it has seen it fail / seen a good skill succeed?

    python -m src.smerl.eval_adapt_multiep --model bc_hgdagger_3skill.pt --skills 0,1,2
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
from src.smerl.collect_trajectories import make_base_env_fn, wrap_value_norm
from src.smerl.eval_bc_multimodal import load_bc
from src.smerl.collect_demos import sample_theta
from src.smerl.adaptive_transformer import AdaptiveTransformer


@torch.no_grad()
def rollout_meta(model, agent, bamdp, device, theta, n_sub, max_steps,
                 teacher_action=False):
    """One meta-episode: n_sub chained sub-episodes under a persistent context.
    Stops early if the running token sequence would exceed the model's context."""
    at = AdaptiveTransformer(model, device)
    max_ctx = model.cfg.n_positions
    bamdp.reset_meta(theta=theta)
    subs = []
    for _ in range(n_sub):
        obs = bamdp.reset()
        active, sk, val, fail = None, [], [], []
        terminated = truncated = False
        info = {}
        t = 0
        while not (terminated or truncated) and t < max_steps:
            if len(at._ids) + 3 > max_ctx:                 # context budget guard
                break
            s = obs["state"].astype(np.float32)
            at.update(s)
            z = at.sample_skill()                          # on-policy argmax
            if active is None:
                active = z; bamdp.set_active_skill(z)
            elif z != active:
                bamdp.switch_skill(z); active = z
                if bamdp.cfg.reset_on_switch:
                    # teleported retry: the skill was decided on the old state; act
                    # from the fresh start and overwrite this step's state token
                    obs = bamdp._observe()
                    s = obs["state"].astype(np.float32)
                    at.revise_state(s)
            v = float(bamdp._v_obs)
            if teacher_action:
                a = np.asarray(agent.act(s, z=active, deterministic=True), np.float32)
                at.skip_action()
            else:
                a = at.sample_action()                     # full policy: transformer action
            at.push_value(v)
            sk.append(active); val.append(v); fail.append(bool(bamdp._failed))
            obs, r, terminated, truncated, info = bamdp.step(a)
            t += 1
        if not sk:
            break
        subs.append({"skills": np.array(sk), "value": np.array(val),
                     "failing": np.array(fail), "success": bool(info.get("is_success")),
                     "first": int(sk[0]), "last": int(sk[-1]),
                     "switched": len(set(sk)) > 1})
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--model", type=str, default="bc_adapt_3skill.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--skills", type=str, default="0,1,2")
    ap.add_argument("--n-meta", type=int, default=60)
    ap.add_argument("--n-sub", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=100,
                    help="must match the value-norm horizon used at collection (>=75)")
    ap.add_argument("--teacher-action", action="store_true",
                    help="drive with the SMERL action for the model's chosen skill "
                         "(isolates skill-switching from the action head)")
    ap.add_argument("--reset-on-switch", action="store_true",
                    help="env-side terminate-on-switch (must match the collection "
                         "dynamics of the model under eval)")
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--theta-mode", choices=["train", "legacy"], default="train",
                    help="train: sample theta exactly like collection (sample_theta — "
                         "one forced-bad, one forced-good, remaining play skills "
                         "~Beta, so e.g. 2-bad-1-good occurs); legacy: one bad at "
                         "--theta-bad, ALL other play skills at --theta-good")
    ap.add_argument("--theta-bad", type=float, default=0.95)
    ap.add_argument("--theta-good", type=float, default=0.05)
    ap.add_argument("--beta-a", type=float, default=0.5)
    ap.add_argument("--beta-b", type=float, default=0.5)
    ap.add_argument("--good-thr", type=float, default=0.15,
                    help="train mode: upper bound on the forced good skill's rate")
    ap.add_argument("--n-plot", type=int, default=3)
    ap.add_argument("--out", type=str, default="adapt_multiep.png")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = os.path.join("src/smerl", args.run)
    agent, cfg, nominal_start, _ = load_agent(os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    goal = build_env(cfg).goal
    nominal_start = np.asarray(nominal_start, np.float32)
    n_skills = cfg["n_skills"]
    play = [int(s) for s in args.skills.split(",")]
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    value_net, _, _ = wrap_value_norm(value_net, play, agent, cfg, nominal_start,
                                      args.radius, device, n_skills)
    model = load_bc(os.path.join(run_dir, args.model), device)
    base_env_fn = make_base_env_fn(cfg, nominal_start, goal, args.radius, args.max_steps)

    cfg_b = BAMDPConfig(n_skills=n_skills, skill_lengths=np.full(n_skills, float(args.max_steps)),
                        n_sub_episodes=10**9, schedule="budget", continue_on_failure=True,
                        expose_value=True, expose_failing=True, plateau_prob=0.5,
                        certain_skill=True, reset_on_switch=args.reset_on_switch)
    bamdp = SyntheticFailureBAMDP(base_env_fn, agent.disc, cfg_b,
                                  np.random.default_rng(args.seed), device, value_net=value_net)

    rng = np.random.default_rng(args.seed)
    succ_by_idx = np.zeros(args.n_sub); cnt_by_idx = np.zeros(args.n_sub)
    badfirst_by_idx = np.zeros(args.n_sub)
    n_switch_when_bad = n_bad_committed = 0
    examples = []
    n_extra_bad = 0          # train mode: meta-eps where a non-forced skill is also bad
    # breakdown by # of bad play skills (theta > 0.5): nbad -> [succ, cnt] per sub-idx
    succ_by_nbad = np.zeros((len(play) + 1, args.n_sub))
    cnt_by_nbad = np.zeros((len(play) + 1, args.n_sub))
    meta_by_nbad = np.zeros(len(play) + 1, dtype=int)
    for m in range(args.n_meta):
        if args.theta_mode == "train":
            theta, bad, good = sample_theta(rng, n_skills, play, args.beta_a,
                                            args.beta_b, args.theta_bad, args.good_thr)
            n_extra_bad += int(any(theta[z] > 0.5 for z in play if z not in (bad, good)))
        else:
            bad = int(rng.choice(play))
            theta = np.full(n_skills, args.theta_bad)
            for z in play:
                if z != bad:
                    theta[z] = args.theta_good
        nbad = int(sum(theta[z] > 0.5 for z in play))
        meta_by_nbad[nbad] += 1
        bamdp.rng = np.random.default_rng(args.seed + 1 + m)
        subs = rollout_meta(model, agent, bamdp, device, theta, args.n_sub,
                            args.max_steps, teacher_action=args.teacher_action)
        for i, sub in enumerate(subs):
            cnt_by_idx[i] += 1
            succ_by_idx[i] += int(sub["success"])
            badfirst_by_idx[i] += int(sub["first"] == bad)
            cnt_by_nbad[nbad, i] += 1
            succ_by_nbad[nbad, i] += int(sub["success"])
            if sub["first"] == bad:                      # committed to the bad skill
                n_bad_committed += 1
                n_switch_when_bad += int(sub["last"] != bad)
        if len(examples) < args.n_plot:
            examples.append((bad, subs))

    print(f"[adapt] {args.n_meta} meta-eps x {args.n_sub} sub-eps  (bad skill random in "
          f"{play}; theta-mode={args.theta_mode})")
    if args.theta_mode == "train" and len(play) > 2:
        print(f"  meta-eps where a non-forced play skill is ALSO bad (theta>0.5): "
              f"{n_extra_bad}/{args.n_meta} = {n_extra_bad/args.n_meta:.2f}")
    print("  success rate by sub-episode index (latching -> rises):")
    print("   " + "  ".join(f"#{i}:{succ_by_idx[i]/max(cnt_by_idx[i],1):.2f}"
                            for i in range(args.n_sub)))
    print("  P(first skill == bad) by sub-episode index (avoidance -> falls):")
    print("   " + "  ".join(f"#{i}:{badfirst_by_idx[i]/max(cnt_by_idx[i],1):.2f}"
                            for i in range(args.n_sub)))
    print(f"  switch-off-bad when committed to bad: {n_switch_when_bad}/{n_bad_committed} "
          f"= {n_switch_when_bad/max(n_bad_committed,1):.2f}")
    print("  success rate by sub-episode, split by # bad play skills:")
    for nb in range(len(play) + 1):
        if meta_by_nbad[nb] == 0:
            continue
        rates = "  ".join(f"#{i}:{succ_by_nbad[nb, i]/max(cnt_by_nbad[nb, i],1):.2f}"
                          for i in range(args.n_sub))
        print(f"   {nb} bad ({meta_by_nbad[nb]} metas):  {rates}")

    # success-by-sub-episode curve, one line per # of bad skills (+ overall)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = np.arange(args.n_sub)
    ax.plot(xs, succ_by_idx / np.maximum(cnt_by_idx, 1), "k--", lw=1.5, marker="o",
            label=f"all ({args.n_meta} metas)")
    for nb in range(len(play) + 1):
        if meta_by_nbad[nb] == 0:
            continue
        rate = succ_by_nbad[nb] / np.maximum(cnt_by_nbad[nb], 1)
        se = np.sqrt(rate * (1 - rate) / np.maximum(cnt_by_nbad[nb], 1))
        ax.errorbar(xs, rate, yerr=se, lw=2, marker="o", capsize=3,
                    label=f"{nb} bad skill{'s' if nb != 1 else ''} ({meta_by_nbad[nb]} metas)")
    ax.set_xticks(xs); ax.set_xlabel("sub-episode index")
    ax.set_ylabel("success rate"); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title(f"{os.path.basename(args.model)} — success by sub-episode "
                 f"(theta-mode={args.theta_mode})", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    curve_out = os.path.join(run_dir, os.path.splitext(args.out)[0] + "_bynbad.png")
    plt.savefig(curve_out, dpi=130)
    print(f"[viz] saved {curve_out}")

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 2.6 * len(examples)),
                             squeeze=False)
    for r, (bad, subs) in enumerate(examples):
        ax = axes[r, 0]; x0 = 0
        for i, sub in enumerate(subs):
            xs = np.arange(x0, x0 + len(sub["value"]))
            ax.plot(xs, sub["value"], color="0.7", lw=1, zorder=1)
            ax.scatter(xs, sub["value"], c=[cmap(z) for z in sub["skills"]], s=12, zorder=2)
            ax.axvline(x0 - 0.5, color="k", ls=":", lw=0.8, alpha=0.5)
            ax.text(x0 + 1, 1.03, f"#{i}{'✓' if sub['success'] else '✗'}", fontsize=7)
            x0 += len(sub["value"])
        ax.set_ylim(-0.02, 1.12); ax.set_ylabel("value")
        ax.set_title(f"meta-ep (bad skill = {bad}); color = committed skill", fontsize=10)
    for z in play:
        axes[0, 0].plot([], [], color=cmap(z), lw=3, label=f"skill {z}")
    axes[0, 0].legend(fontsize=8, ncol=len(play), loc="lower right")
    axes[-1, 0].set_xlabel("timestep (sub-episodes concatenated; : = reset)")
    plt.tight_layout()
    out = os.path.join(run_dir, args.out)
    plt.savefig(out, dpi=130)
    print(f"[viz] saved {out}")


if __name__ == "__main__":
    main()
