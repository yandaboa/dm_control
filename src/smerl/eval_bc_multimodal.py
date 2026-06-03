"""Closed-loop evaluation of the multimodal skill-token BC transformer.

The model was trained on  [s_0, z_0, v_0, s_1, z_1, v_1, ...]  to predict
z_t (categorical) from the state token and a_t (gaussian) from the skill token,
with v_t = V(s_t, z_t) fed as context. Reordering z before v lets us run it in
the loop without circularity:

    at step t:  append s_t  -> read skill logits off s_t's hidden -> pick z_t
                append z_t  -> read action off z_t's hidden       -> a_t
                v_t = V(s_t, z_t)  -> append as context;  step env with a_t

Two evals:
  * forced-skill : clamp z_t = z for the whole episode, for each z. Measures
    whether the single transformer reproduces every teacher skill (success/dist
    vs the SMERL teacher for that skill).
  * sampled      : sample z_t ~ p(.|s_t). Reports the distribution of committed
    skills, the within-episode commitment (fraction of steps on the modal skill),
    and success. Multimodality = different episodes from the same start commit to
    different skills; commitment = the skill stays put within an episode.

    python -m src.smerl.eval_bc_multimodal --run runs/smerl_lowalpha_ckpt \
        --model bc_multimodal.pt
"""

from __future__ import annotations

import argparse
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
                          head=blob.get("head", "mse"),
                          discrete=blob.get("discrete"),
                          head_types=blob.get("head_types"),
                          loss_weights=blob.get("loss_weights"),
                          n_bins=int(blob.get("n_bins", 21)),
                          logic=bool(blob.get("logic", False))).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def rollout_multimodal(model, value_net, env, device, force_z=None,
                       temperature=1.0, rng=None):
    """Autoregressive rollout. If force_z is None, sample z_t ~ p(.|s_t);
    otherwise clamp z_t = force_z. Returns (success, ret, len, dist, states,
    z_seq) where z_seq is the per-step committed skill."""
    sid, zid, vid = model.id_of["state"], model.id_of["skill"], model.id_of["value"]
    D = model.max_dim
    rng = rng or np.random.default_rng(0)
    obs, _ = env.reset()
    ids, vals = [], []
    states, z_seq = [], []
    ret, t = 0.0, 0
    terminated = truncated = False
    info = {}

    def _emb_forward():
        tid = torch.as_tensor(ids, device=device)[None]
        tval = torch.as_tensor(np.stack(vals), device=device)[None]
        attn = torch.ones_like(tid, dtype=torch.float32)
        return model.backbone(model.embed(tid, tval), attn)

    while not (terminated or truncated):
        s = np.asarray(obs, dtype=np.float32)
        sv = np.zeros(D, np.float32); sv[:len(s)] = s
        ids.append(sid); vals.append(sv)
        # skill logits off the state token's hidden state
        hidden = _emb_forward()
        logits = model.head_logits("skill", hidden[0, -1]).cpu().numpy()
        if force_z is None:
            p = np.exp((logits - logits.max()) / temperature)
            p = p / p.sum()
            z = int(rng.choice(len(p), p=p))
        else:
            z = int(force_z)
        states.append(s); z_seq.append(z)
        # append skill token, read action off its hidden state
        zv = np.zeros(D, np.float32); zv[0] = z
        ids.append(zid); vals.append(zv)
        hidden = _emb_forward()
        a = torch.clamp(model.head_mean("action", hidden[0, -1]), -1, 1).cpu().numpy()
        # value token v_t = V(s_t, z_t) as context
        v = float(value_net(torch.as_tensor(s[None], device=device),
                            torch.tensor([z], device=device)).item())
        vv = np.zeros(D, np.float32); vv[0] = v
        ids.append(vid); vals.append(vv)
        obs, r, terminated, truncated, info = env.step(a)
        ret += float(r); t += 1
    return (bool(info.get("is_success", False)), ret, t,
            float(info.get("distance_to_goal", np.nan)), np.asarray(states),
            np.asarray(z_seq, dtype=int))


@torch.no_grad()
def rollout_teacher(agent, env, z):
    obs, _ = env.reset()
    states = [np.asarray(obs, np.float32)]
    ret, t = 0.0, 0
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        a = agent.act(obs, z=z, deterministic=True)
        obs, r, terminated, truncated, info = env.step(a)
        states.append(np.asarray(obs, np.float32))
        ret += float(r); t += 1
    return (bool(info.get("is_success", False)), ret, t,
            float(info.get("distance_to_goal", np.nan)), np.asarray(states))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--model", type=str, default="bc_multimodal.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=75)
    ap.add_argument("--temperature", type=float, default=1.0)
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

    # fixed start set: every policy sees identical episodes
    starts = [sample_in_disk(nominal_start, args.radius,
                             np.random.default_rng(args.seed + i))
              for i in range(args.n_episodes)]

    def run(fn):
        out = [fn(build_env(cfg, start=tuple(st)), i) for i, st in enumerate(starts)]
        return out

    def summ(rs):   # rs: list of (succ, ret, len, dist, ...)
        return (np.mean([r[0] for r in rs]), np.mean([r[1] for r in rs]),
                np.mean([r[2] for r in rs]), np.mean([r[3] for r in rs]))

    # --- per-skill: teacher vs forced-skill BC ---
    print(f"{'skill':>5}  {'teacher succ/dist':>20}  {'BC(forced) succ/dist':>22}")
    rows = []
    for z in range(n_skills):
        ts, tr_, tl, td = summ(run(lambda e, i, z=z: rollout_teacher(agent, e, z)))
        bs, br, bl, bd = summ(run(
            lambda e, i, z=z: rollout_multimodal(model, value_net, e, device,
                                                 force_z=z)))
        rows.append((z, ts, td, bs, bd))
        print(f"{z:>5}  {ts:>10.3f}/{td:<8.3f}  {bs:>11.3f}/{bd:<8.3f}")

    # --- sampled: commitment + multimodality ---
    samp = run(lambda e, i: rollout_multimodal(
        model, value_net, e, device, force_z=None, temperature=args.temperature,
        rng=np.random.default_rng(1000 + i)))
    ssucc, _, slen, sdist = summ(samp)
    committed = np.array([np.bincount(r[5], minlength=n_skills).argmax()
                          for r in samp])
    commit_frac = np.array([np.mean(r[5] == np.bincount(r[5],
                            minlength=n_skills).argmax()) for r in samp])
    skill_hist = np.bincount(committed, minlength=n_skills)
    print(f"\n[sampled]  success={ssucc:.3f}  len={slen:.1f}  dist={sdist:.3f}  "
          f"temp={args.temperature}")
    print(f"[sampled]  committed-skill histogram: {skill_hist.tolist()} "
          f"(of {args.n_episodes})")
    print(f"[sampled]  within-episode commitment (mean frac on modal skill): "
          f"{commit_frac.mean():.3f}")

    # --- plot ---
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    zs = np.arange(n_skills)
    w = 0.38
    ax[0].bar(zs - w / 2, [r[1] for r in rows], w, label="teacher", color="0.6")
    ax[0].bar(zs + w / 2, [r[3] for r in rows], w, label="BC forced", color="#1f77b4")
    ax[0].set_xticks(zs); ax[0].set_xlabel("skill z"); ax[0].set_ylim(0, 1.02)
    ax[0].set_ylabel("success rate"); ax[0].set_title("Per-skill: teacher vs forced BC")
    ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].bar(zs, skill_hist, color="#2ca02c")
    ax[1].set_xticks(zs); ax[1].set_xlabel("committed skill")
    ax[1].set_ylabel("# episodes"); ax[1].grid(alpha=.3)
    ax[1].set_title(f"Sampled commitment ({args.n_episodes} eps from start disk)\n"
                    f"commitment={commit_frac.mean():.2f}  success={ssucc:.2f}")

    cmap = plt.get_cmap("tab10")
    goal = build_env(cfg).goal
    for r in samp:
        st, zc = r[4], int(np.bincount(r[5], minlength=n_skills).argmax())
        ax[2].plot(st[:, 0], st[:, 1], color=cmap(zc), alpha=0.25, lw=0.8)
    for z in range(n_skills):
        ax[2].plot([], [], color=cmap(z), label=f"z={z}")
    ax[2].scatter(*nominal_start, marker="*", c="k", s=200, zorder=5)
    ax[2].scatter(goal[0], goal[1], marker="X", c="k", s=160, zorder=5)
    ax[2].set_xlim(-1, 1); ax[2].set_ylim(-1, 1); ax[2].set_aspect("equal")
    ax[2].set_title("Sampled trajectories (colored by committed skill)")
    ax[2].legend(fontsize=8, loc="upper left"); ax[2].grid(alpha=.3)

    fig.suptitle(f"Multimodal skill-token BC — {args.model} "
                 f"({args.n_episodes} eps, start∈disk r={args.radius}, T<={args.max_steps})")
    plt.tight_layout()
    out = os.path.join(run_dir, "bc_multimodal_eval.png")
    plt.savefig(out, dpi=130)
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
