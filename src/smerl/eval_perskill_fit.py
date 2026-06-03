"""Diagnose per-skill BC failure: expressiveness vs covariate shift.

For each skill z we separate two questions:

  1. Can the model REPRESENT skill z's policy mean?  (expressiveness)
     -> teacher-force the model on the demo sequences and compare its predicted
        mean action to the teacher's DETERMINISTIC action at the same state
        (removes the aleatoric-variance confound in NLL). Low = expressive.

  2. Does closed-loop rollout leave the demo distribution?  (covariate shift)
     -> roll out forced-z, then for each visited state measure (a) its distance
        to the nearest demo state for skill z (how far off-manifold we drift) and
        (b) the model-mean-vs-teacher action error at those visited states.

If on-distribution mean-fit is uniformly low but a skill's rollout drifts far
off-manifold with exploding action error, the failure is compounding/covariate
shift, NOT capacity.

    python -m src.smerl.eval_perskill_fit --run runs/smerl_lowalpha_ckpt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env, sample_in_disk
from src.smerl.train_task_value import load_value_net
from src.smerl.seq_data import SeqDataset, SequenceSpec
from src.smerl.eval_bc_multimodal import load_bc, rollout_multimodal


@torch.no_grad()
def teacher_forced_means(model, ep, device):
    """Run the model on one demo episode's token sequence; return predicted mean
    actions at the skill-token positions, aligned with timesteps t=0..T-1."""
    toks = model_seq_tokens(model, ep)
    ids = torch.as_tensor([t[0] for t in toks], device=device)[None]
    D = model.max_dim
    vals = np.zeros((len(toks), D), np.float32)
    for j, t in enumerate(toks):
        vals[j, :len(t[1])] = t[1]
    vals = torch.as_tensor(vals, device=device)[None]
    attn = torch.ones_like(ids, dtype=torch.float32)
    hidden = model.backbone(model.embed(ids, vals), attn)[0]
    zpos = (ids[0] == model.id_of["skill"]).nonzero(as_tuple=True)[0]
    means = model.head_mean("action", hidden[zpos])
    return torch.clamp(means, -1, 1).cpu().numpy()


def model_seq_tokens(model, ep):
    """Rebuild the [s,z,v] input tokens for an episode (ids, raw vec)."""
    spec = SequenceSpec(pattern=[["state", "skill"], ["skill", "action"], "value"])
    out = []
    for in_name, in_vec, _, _ in spec.tokens(ep):
        out.append((model.id_of[in_name], in_vec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--model", type=str, default="bc_multimodal.pt")
    ap.add_argument("--store", type=str, default="trajectories_multimodal")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--n-roll", type=int, default=60, help="rollouts/skill")
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=75)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = os.path.join("src/smerl", args.run)
    agent, cfg, nominal_start, _ = load_agent(
        os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    nominal_start = np.asarray(nominal_start, np.float32)
    n_skills = cfg["n_skills"]
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    model = load_bc(os.path.join(run_dir, args.model), device)

    store = os.path.join(run_dir, args.store)
    spec = SequenceSpec(pattern=[["state", "skill"], ["skill", "action"], "value"])
    ds = SeqDataset(store, spec)

    # group demo episodes + states by skill
    demo_states = {z: [] for z in range(n_skills)}
    onfit = {z: [] for z in range(n_skills)}   # |model_mean - teacher_det|^2
    from src.smerl.trajectory_store import load_episode
    for rec in ds.records:
        ep = load_episode(store, rec)
        z = int(ep["z"])
        S = ep["states"][:-1]                  # states aligned with actions
        demo_states[z].append(S.astype(np.float32))
        means = teacher_forced_means(model, ep, device)   # [T, act_dim]
        tdet = np.stack([agent.act(s, z=z, deterministic=True) for s in S])
        m = min(len(means), len(tdet))
        onfit[z].append(((means[:m] - tdet[:m]) ** 2).sum(-1))
    demo_states = {z: np.concatenate(v) for z, v in demo_states.items()}

    rng = np.random.default_rng(args.seed)

    def nearest_dist(pts, ref, cap=3000):
        """mean nearest-neighbour distance of pts to ref set (subsampled)."""
        if len(ref) > cap:
            ref = ref[rng.choice(len(ref), cap, replace=False)]
        d = np.sqrt(((pts[:, None, :] - ref[None, :, :]) ** 2).sum(-1))
        return d.min(1)

    print(f"{'z':>2} {'on-fit MSE':>11} {'succ':>6} {'rollMSE':>8} "
          f"{'driftNN':>8} {'demoNN':>7}")
    rows = []
    for z in range(n_skills):
        onfit_mse = float(np.concatenate(onfit[z]).mean())
        # closed-loop forced-z rollouts
        roll_states, roll_err, succ = [], [], []
        for i in range(args.n_roll):
            st = sample_in_disk(nominal_start, args.radius,
                                np.random.default_rng(args.seed + i))
            env = build_env(cfg, start=tuple(st))
            ok, _, _, _, states, _ = rollout_multimodal(
                model, value_net, env, device, force_z=z)
            succ.append(ok)
            if len(states):
                tdet = np.stack([agent.act(s, z=z, deterministic=True)
                                 for s in states])
                mean = teacher_forced_action_at(model, value_net, states, z, device)
                roll_err.append(((mean - tdet) ** 2).sum(-1))
                roll_states.append(states)
        roll_states = np.concatenate(roll_states)
        roll_err = float(np.concatenate(roll_err).mean())
        drift = float(nearest_dist(roll_states, demo_states[z]).mean())
        # baseline: how spread are the demo states themselves (NN within demos)
        demo_nn = float(nearest_dist(
            demo_states[z][rng.choice(len(demo_states[z]),
                                      min(1500, len(demo_states[z])), replace=False)],
            demo_states[z]).mean())
        s = float(np.mean(succ))
        rows.append((z, onfit_mse, s, roll_err, drift, demo_nn))
        print(f"{z:>2} {onfit_mse:>11.4f} {s:>6.2f} {roll_err:>8.4f} "
              f"{drift:>8.4f} {demo_nn:>7.4f}")
    print("\nLegend: on-fit MSE = |model mean - teacher det| on DEMO states "
          "(expressiveness, lower=more expressive)")
    print("        rollMSE = same error but on CLOSED-LOOP states; "
          "driftNN = mean dist of rollout states to nearest demo state; "
          "demoNN = within-demo NN dist (drift baseline)")


@torch.no_grad()
def teacher_forced_action_at(model, value_net, states, z, device):
    """Re-run the model causally over a state sequence with skill forced to z,
    returning the predicted mean action at each step (the action actually used in
    a forced-z rollout)."""
    sid, zid, vid = model.id_of["state"], model.id_of["skill"], model.id_of["value"]
    D = model.max_dim
    ids, vals, means = [], [], []
    for s in states:
        sv = np.zeros(D, np.float32); sv[:len(s)] = s
        ids.append(sid); vals.append(sv)
        zv = np.zeros(D, np.float32); zv[0] = z
        ids.append(zid); vals.append(zv)
        tid = torch.as_tensor(ids, device=device)[None]
        tval = torch.as_tensor(np.stack(vals), device=device)[None]
        attn = torch.ones_like(tid, dtype=torch.float32)
        hidden = model.backbone(model.embed(tid, tval), attn)
        a = torch.clamp(model.head_mean("action", hidden[0, -1]), -1, 1).cpu().numpy()
        means.append(a)
        v = float(value_net(torch.as_tensor(s[None], device=device),
                            torch.tensor([z], device=device)).item())
        vv = np.zeros(D, np.float32); vv[0] = v
        ids.append(vid); vals.append(vv)
    return np.stack(means)


if __name__ == "__main__":
    main()
