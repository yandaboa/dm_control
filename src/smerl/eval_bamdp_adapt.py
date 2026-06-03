"""Closed-loop ADAPTATION eval in the synthetic-failure BAMDP.

The real test of DAgger iteration 1: drop the multimodal transformer into the
failure BAMDP and let it drive itself. Each step it predicts a skill from the
state token (which causally sees the value history); when it changes its skill we
call ``bamdp.switch_skill`` — re-evaluating failure under the new skill while
still allowing future failures (the new skill may also be bad). A doomed skill's
value stagnates; a transformer that learned the intervention behavior should
abandon it and commit to a good one, recovering to the goal.

We report, per model, the success rate overall and — the metric that matters —
success on the episodes where a failure was actually injected (i.e. adaptation
was required), plus the mean number of skill switches.

    python -m src.smerl.eval_bamdp_adapt \
        --models bc_multimodal.pt,bc_multimodal_dagger.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env
from src.smerl.train_task_value import load_value_net
from src.smerl.bamdp_env import BAMDPConfig, SyntheticFailureBAMDP
from src.smerl.collect_trajectories import make_base_env_fn, wrap_value_norm
from src.smerl.eval_bc_multimodal import load_bc


@torch.no_grad()
def rollout_adapt(model, bamdp, device, sample=False, temperature=1.0, rng=None):
    """Drive the transformer in the BAMDP, switching skills on its own."""
    sid, zid, vid = model.id_of["state"], model.id_of["skill"], model.id_of["value"]
    D = model.max_dim
    obs = bamdp.reset()
    ids, vals = [], []
    ret = t = switches = 0
    any_fail = False
    active = None
    terminated = truncated = False
    info = {}

    def fwd():
        tid = torch.as_tensor(ids, device=device)[None]
        tval = torch.as_tensor(np.stack(vals), device=device)[None]
        attn = torch.ones_like(tid, dtype=torch.float32)
        return model.backbone(model.embed(tid, tval), attn)

    while not (terminated or truncated):
        s = obs["state"].astype(np.float32)
        sv = np.zeros(D, np.float32); sv[:len(s)] = s
        ids.append(sid); vals.append(sv)
        logits = model.head_logits("skill", fwd()[0, -1]).cpu().numpy()
        if sample:
            p = np.exp((logits - logits.max()) / temperature); p /= p.sum()
            z = int((rng or np.random).choice(len(p), p=p))
        else:
            z = int(logits.argmax())
        if active is None:
            active = z; bamdp.set_active_skill(z)
        elif z != active:
            bamdp.switch_skill(z); active = z; switches += 1
            if bamdp.cfg.reset_on_switch:
                # the skill decision used the (stalled) pre-teleport context, but
                # the action must be taken from the fresh retry start: re-observe
                # and overwrite this step's state token with the teleported state
                obs = bamdp._observe()
                s = obs["state"].astype(np.float32)
                sv = np.zeros(D, np.float32); sv[:len(s)] = s
                vals[-1] = sv
        v = float(bamdp._v_obs)                  # exposed value under current skill
        zv = np.zeros(D, np.float32); zv[0] = z
        ids.append(zid); vals.append(zv)
        a = torch.clamp(model.head_mean("action", fwd()[0, -1]), -1, 1).cpu().numpy()
        vv = np.zeros(D, np.float32); vv[0] = v
        ids.append(vid); vals.append(vv)
        obs, r, terminated, truncated, info = bamdp.step(a)
        any_fail = any_fail or bool(info["forced_failure"])
        ret += float(r); t += 1
    return {"success": bool(info.get("is_success", False)), "switches": switches,
            "any_fail": any_fail, "final_skill": active, "len": t,
            "theta": bamdp.theta.copy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--models", type=str,
                    default="bc_multimodal.pt,bc_multimodal_dagger.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--beta-a", type=float, default=0.5)
    ap.add_argument("--beta-b", type=float, default=0.5)
    ap.add_argument("--sample", action="store_true",
                    help="sample the skill each step (default: argmax)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--skills", type=str, default=None,
                    help="restrict the latent to these skills (one made bad/episode)")
    ap.add_argument("--reset-on-switch", action="store_true")
    ap.add_argument("--theta-bad", type=float, default=0.95)
    ap.add_argument("--value-norm-skills", type=str, default=None,
                    help="per-skill renormalize V(s,z)->[0,1] to match training")
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = os.path.join("src/smerl", args.run)
    agent, cfg, nominal_start, _ = load_agent(
        os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    goal = build_env(cfg).goal
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    n_skills = cfg["n_skills"]
    base_env_fn = make_base_env_fn(cfg, nominal_start, goal, args.radius,
                                   args.max_steps)
    if args.value_norm_skills:                # match the (normalized) training value
        ns = [int(s) for s in args.value_norm_skills.split(",")]
        value_net, _, _ = wrap_value_norm(value_net, ns, agent, cfg, nominal_start,
                                          args.radius, device, n_skills)
    cfg_b = BAMDPConfig(n_skills=n_skills,
                        skill_lengths=np.full(n_skills, float(args.max_steps)),
                        n_sub_episodes=10**9, schedule="budget",
                        beta_a=args.beta_a, beta_b=args.beta_b,
                        continue_on_failure=True, expose_value=True,
                        expose_failing=True, certain_skill=True,
                        reset_on_switch=args.reset_on_switch)
    bamdp = SyntheticFailureBAMDP(base_env_fn, agent.disc, cfg_b,
                                  np.random.default_rng(0), device,
                                  value_net=value_net)
    allowed = ([int(s) for s in args.skills.split(",")] if args.skills else None)

    def set_task(i):
        bamdp.rng = np.random.default_rng(args.seed + i)       # same task per model
        if allowed is not None:                                # 2-skill: one bad
            bad = int(bamdp.rng.choice(allowed))
            theta = np.full(n_skills, 0.02); theta[bad] = args.theta_bad
            bamdp.reset_meta(theta=theta)
        else:
            bamdp.reset_meta()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"{'model':>26} {'success':>8} {'succ|failed':>12} "
          f"{'%failed':>8} {'switches':>9}")
    for mname in models:
        model = load_bc(os.path.join(run_dir, mname), device)
        rows = []
        for i in range(args.n_episodes):
            set_task(i)
            rows.append(rollout_adapt(model, bamdp, device, sample=args.sample,
                                      temperature=args.temperature,
                                      rng=np.random.default_rng(1000 + i)))
        succ = np.mean([r["success"] for r in rows])
        failed = [r for r in rows if r["any_fail"]]
        succ_failed = (np.mean([r["success"] for r in failed]) if failed
                       else float("nan"))
        sw = np.mean([r["switches"] for r in rows])
        print(f"{mname:>26} {succ:>8.3f} {succ_failed:>12.3f} "
              f"{len(failed)/len(rows):>8.2f} {sw:>9.2f}")


if __name__ == "__main__":
    main()
