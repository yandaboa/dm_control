"""Closed-loop rollout evaluation of the BC GPT2 models from the data ablation.

Each model was trained on [s_1, v_1, ..., s_t, v_t] -> predict a_t from v_t's
hidden state. Here we run it *in the loop*: at each step we compute the
skill-conditioned value v_t = V(s_t, z=0), append [s_t, v_t] to the causal
context, read the action off the last (value) token, step the env, and recurse.
We report success rate / return / length over many starts, against the original
SMERL skill-0 policy as the teacher reference.

    python -m src.smerl.eval_bc_rollout --run runs/smerl_lowalpha_ckpt \
        --ckpt-file ckpt_step010000.pt
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.smerl.skill_decode import load_agent, build_env, sample_in_disk
from src.smerl.train_task_value import load_value_net
from src.smerl.seq_model import TrajectoryGPT, GPT2Config


def load_bc(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = TrajectoryGPT(blob["modalities"], GPT2Config(**blob["cfg"]),
                          head=blob.get("head", "mse")).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def rollout_bc(model, value_net, env, device, z=0):
    """Autoregressive closed-loop rollout of the BC model in env."""
    state_id, value_id = model.id_of["state"], model.id_of["value"]
    D = model.max_dim
    obs, _ = env.reset()
    ids, vals = [], []
    ret, t = 0.0, 0
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        s = np.asarray(obs, dtype=np.float32)
        v = float(value_net(torch.as_tensor(s[None], device=device),
                            torch.tensor([z], device=device)).item())
        sv = np.zeros(D, np.float32); sv[:len(s)] = s
        vv = np.zeros(D, np.float32); vv[0] = v
        ids += [state_id, value_id]; vals += [sv, vv]
        tid = torch.as_tensor(ids, device=device)[None]
        tval = torch.as_tensor(np.stack(vals), device=device)[None]
        attn = torch.ones_like(tid, dtype=torch.float32)
        hidden = model.backbone(model.embed(tid, tval), attn)
        a = torch.clamp(model.head_mean("action", hidden[0, -1]), -1, 1).cpu().numpy()
        obs, r, terminated, truncated, info = env.step(a)
        ret += float(r); t += 1
    return bool(info.get("is_success", False)), ret, t, \
        float(info.get("distance_to_goal", np.nan))


@torch.no_grad()
def rollout_teacher(agent, env, z=0):
    obs, _ = env.reset()
    ret, t = 0.0, 0
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        a = agent.act(obs, z=z, deterministic=True)
        obs, r, terminated, truncated, info = env.step(a)
        ret += float(r); t += 1
    return bool(info.get("is_success", False)), ret, t, \
        float(info.get("distance_to_goal", np.nan))


def summarize(results):
    succ = np.mean([r[0] for r in results])
    ret = np.mean([r[1] for r in results])
    ln = np.mean([r[2] for r in results])
    dist = np.mean([r[3] for r in results])
    return succ, ret, ln, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--models", type=str, default="bc_skill0_n*.pt",
                    help="glob (within run dir) of BC checkpoints")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=75)
    ap.add_argument("--z", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dir = os.path.join("src/smerl", args.run)
    agent, cfg, nominal_start, _ = load_agent(
        os.path.join(run_dir, args.ckpt_file), device)
    cfg["max_episode_steps"] = args.max_steps
    goal = build_env(cfg).goal
    nominal_start = np.asarray(nominal_start, np.float32)
    value_net, _ = load_value_net(os.path.join(run_dir, "value_net_skill.pt"),
                                  device)

    def make_env(rng):
        start = sample_in_disk(nominal_start, args.radius, rng)
        return build_env(cfg, start=tuple(start))

    # fixed start set so every model + teacher see identical episodes
    starts = [sample_in_disk(nominal_start, args.radius,
                             np.random.default_rng(args.seed + i))
              for i in range(args.n_episodes)]

    def run_policy(fn):
        out = []
        for st in starts:
            env = build_env(cfg, start=tuple(st))
            out.append(fn(env))
        return summarize(out)

    # teacher reference
    tsucc, tret, tln, tdist = run_policy(lambda e: rollout_teacher(agent, e, args.z))
    print(f"[teacher z={args.z}]  success={tsucc:.3f}  return={tret:.1f}  "
          f"len={tln:.1f}  final_dist={tdist:.3f}")

    paths = sorted(glob.glob(os.path.join(run_dir, args.models)),
                   key=lambda p: int("".join(c for c in os.path.basename(p)
                                             if c.isdigit())))
    rows = []
    for p in paths:
        n = int("".join(c for c in os.path.basename(p) if c.isdigit()))
        model = load_bc(p, device)
        s, ret, ln, dist = run_policy(
            lambda e: rollout_bc(model, value_net, e, device, args.z))
        rows.append((n, s, ret, ln, dist))
        print(f"[BC n={n:5d}]  success={s:.3f}  return={ret:.1f}  len={ln:.1f}  "
              f"final_dist={dist:.3f}")

    # plot success + final-distance vs data scale
    ns = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].axhline(tsucc, ls="--", color="k", label=f"teacher ({tsucc:.2f})")
    ax[0].plot(ns, [r[1] for r in rows], "o-", color="#1f77b4")
    ax[0].set_xscale("log"); ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_xlabel("# demos (log)"); ax[0].set_ylabel("closed-loop success rate")
    ax[0].set_title("BC success vs data scale"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].axhline(tdist, ls="--", color="k", label=f"teacher ({tdist:.3f})")
    ax[1].plot(ns, [r[4] for r in rows], "o-", color="#d62728")
    ax[1].set_xscale("log"); ax[1].set_xlabel("# demos (log)")
    ax[1].set_ylabel("mean final distance to goal")
    ax[1].set_title("BC final distance vs data scale"); ax[1].grid(alpha=.3)
    ax[1].legend()
    fig.suptitle(f"Closed-loop BC eval (skill {args.z}, {args.n_episodes} eps, "
                 f"start∈disk r={args.radius}, T<={args.max_steps})")
    plt.tight_layout()
    out = os.path.join(run_dir, "bc_rollout_eval.png")
    plt.savefig(out, dpi=130)
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
