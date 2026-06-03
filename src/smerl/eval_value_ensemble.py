"""Sanity check for the max-of-ensemble value token.

We want a *skill-agnostic* value to feed the transformer's V token without
leaking which skill a demo came from. The proposal: take the skill-conditioned
ensemble V(s, z) and reduce over z by a max,

    V_max(s) = max_z V(s, z).

For this to be a faithful stand-in for "how good is this state under the skill
we're actually running", the argmax skill at a state visited by skill z should
*be* z (the executing skill values the state highest), and then V_max(s) equals
the correct V(s, z). This script measures both, per skill, over stochastic
rollouts from perturbed starts:

  * argmax-match rate :  mean[ argmax_z' V(s,z') == z_true ]
  * |V_max - V_correct|: mean abs gap between the ensemble max and V(s, z_true)

    python -m src.smerl.eval_value_ensemble --run runs/smerl_lowalpha_ckpt \
        --ckpt-file ckpt_step010000.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env, sample_in_disk
from src.smerl.train_task_value import load_value_net


@torch.no_grad()
def ensemble_values(net, X, n_skills, device):
    """X [N,obs_dim] -> V [N,n_skills] with V[:,z] = V(s, z)."""
    Xt = torch.as_tensor(X, device=device)
    cols = []
    for z in range(n_skills):
        zc = torch.full((Xt.shape[0],), z, dtype=torch.long, device=device)
        cols.append(net(Xt, zc))
    return torch.stack(cols, dim=1).cpu().numpy()     # [N, n_skills]


def collect_states(agent, cfg, nominal_start, n_skills, n_per_skill, radius,
                   rng, deterministic):
    """Stochastic rollouts per skill; return list of (states[T,obs_dim], z)."""
    rollouts = []
    for z in range(n_skills):
        for _ in range(n_per_skill):
            start = sample_in_disk(nominal_start, radius, rng) if radius > 0 \
                else nominal_start
            env = build_env(cfg, start=tuple(start))
            obs, _ = env.reset()
            states = [obs.copy()]
            terminated = truncated = False
            while not (terminated or truncated):
                a = agent.act(obs, z=z, deterministic=deterministic)
                obs, _, terminated, truncated, _ = env.step(a)
                states.append(obs.copy())
            rollouts.append((np.asarray(states, dtype=np.float32), z))
    return rollouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--n-per-skill", type=int, default=100)
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=75)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    run_dir = os.path.join("src/smerl", args.run)
    agent, cfg, nominal_start, obs_dim = load_agent(
        os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    n_skills = cfg["n_skills"]
    nominal_start = np.asarray(nominal_start, dtype=np.float32)

    net, blob = load_value_net(os.path.join(run_dir, args.value_net), device)
    assert getattr(net, "n_skills", None) == n_skills, \
        "value net must be the skill-conditioned ensemble"

    rollouts = collect_states(agent, cfg, nominal_start, n_skills,
                              args.n_per_skill, args.radius, rng,
                              args.deterministic)

    print(f"[ensemble check] {args.value_net}  ({'det' if args.deterministic else 'stochastic'} "
          f"rollouts, {args.n_per_skill}/skill, r={args.radius}, T<={args.max_steps})")
    print(f"{'skill':>5} {'states':>7} {'argmax==z':>10} "
          f"{'|Vmax-Vz|':>10} {'meanVz':>7} {'meanVmax':>9}")
    tot_states = tot_match = 0
    gap_all = []
    conf = np.zeros((n_skills, n_skills), dtype=np.int64)   # rows=true z, cols=argmax
    for z in range(n_skills):
        states = np.concatenate([s for s, zz in rollouts if zz == z], axis=0)
        V = ensemble_values(net, states, n_skills, device)   # [N, n_skills]
        amax = V.argmax(axis=1)
        vmax = V.max(axis=1)
        vz = V[:, z]
        match = (amax == z)
        gap = np.abs(vmax - vz)
        for a in amax:
            conf[z, a] += 1
        tot_states += len(states); tot_match += int(match.sum())
        gap_all.append(gap)
        print(f"{z:>5} {len(states):>7} {match.mean():>10.3f} "
              f"{gap.mean():>10.4f} {vz.mean():>7.3f} {vmax.mean():>9.3f}")
    gap_all = np.concatenate(gap_all)
    print(f"{'ALL':>5} {tot_states:>7} {tot_match/max(tot_states,1):>10.3f} "
          f"{gap_all.mean():>10.4f}")
    print(f"[overall] argmax-match={tot_match/max(tot_states,1):.3f}  "
          f"|Vmax-Vz|: mean={gap_all.mean():.4f} median={np.median(gap_all):.4f} "
          f"p95={np.percentile(gap_all,95):.4f}")
    print("[confusion] rows=true skill, cols=argmax skill (state counts)")
    for z in range(n_skills):
        row = conf[z] / max(conf[z].sum(), 1)
        print(f"  z={z}: " + " ".join(f"{x:.2f}" for x in row))


if __name__ == "__main__":
    main()
