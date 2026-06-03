"""Ablation: which gradient-balancing tricks are necessary to get switching?

One-factor-at-a-time off the winner bc_logic_dec (decouple + switch_weight 20 +
logic gate). For each model we report two switching metrics, matched across models:

  (1) OFFLINE switch analytic (switch_metrics on the merged dataset): at the temporal
      switch-decision positions, the skill head's argmax==new accuracy, P(new), P(stay).
      Pure teacher-forced read of whether the head learned the switch.

  (2) CLOSED-LOOP wall adaptation: every model is FORCED to start committed to the same
      skill and faces the SAME set of freeze-on-contact walls on that skill's path; we
      report how often it switches off the stuck skill and reaches the goal.

    python -m src.smerl.ablate_switching --force-start 2 --n-trials 60
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.smerl.skill_decode import load_agent, build_env
from src.smerl.train_task_value import load_value_net
from src.smerl.bamdp_env import BAMDPConfig, SyntheticFailureBAMDP
from src.smerl.collect_trajectories import wrap_value_norm
from src.smerl.eval_bc_multimodal import load_bc
from src.smerl.wall_env import WallPoint2DGoalEnv, valid_walls
from src.smerl.visualize_bamdp_switch import rollout_record
from src.smerl.seq_data import SeqDataset, SequenceSpec, collate
from src.smerl.train_seq import switch_metrics


# (label, checkpoint file, decouple, switch_weight, logic) — the ablation grid
RUNS = [
    ("winner            (dec, sw20, logic)", "bc_logic_dec.pt",    True,  20, True),
    ("-decouple         (coup,sw20, logic)", "bc_logic_coup.pt",   False, 20, True),
    ("-switch_weight    (dec, sw1,  logic)", "bc_nosw.pt",         True,  1,  True),
    ("-logic            (dec, sw20, no-lg)", "bc_nologic_dec.pt",  True,  20, False),
]


def closed_loop_wall(model, agent, value_net, cfg, start, goal, sr, n_skills,
                     walls, force_start, max_steps, device):
    """Run the model on a fixed set of walls (forced to start on z_block=force_start).
    Returns (goal_rate, switch_rate, switch_to_spare_rate)."""
    z_block = force_start
    z_spare = [s for s in [int(x) for x in (1, 2)] if s != z_block][0]

    def env_fn(wall):
        return WallPoint2DGoalEnv(start=tuple(start), goal=tuple(goal), wall=wall,
                                  success_radius=sr, max_episode_steps=max_steps)

    STALL = 8                       # a first switch at/after this step is stall-driven
    n_succ = n_switch = n_adapt = 0
    sw_steps = []
    for wall in walls:
        cfg_b = BAMDPConfig(n_skills=n_skills,
                            skill_lengths=np.full(n_skills, float(max_steps)),
                            n_sub_episodes=10**9, schedule="budget",
                            continue_on_failure=True, expose_value=True,
                            expose_failing=True, certain_skill=True,
                            reset_on_switch=True)
        bamdp = SyntheticFailureBAMDP(lambda r, w=wall: env_fn(w), agent.disc, cfg_b,
                                      np.random.default_rng(1), device,
                                      value_net=value_net)
        bamdp.reset_meta(theta=np.zeros(n_skills))           # NO injected failures
        ep = rollout_record(model, bamdp, device, force_first=z_block)
        n_succ += int(ep["success"])
        if ep["switches"]:
            n_switch += 1
            s0 = ep["switches"][0]
            sw_steps.append(s0)
            # genuine adaptation: committed to the stuck skill, stalled, THEN switched
            # to the escape and reached the goal (not an immediate step-1 default-away)
            if s0 >= STALL and ep["success"] and int(ep["skill"][-1]) == z_spare:
                n_adapt += 1
    n = max(len(walls), 1)
    mean_sw = float(np.mean(sw_steps)) if sw_steps else 0.0
    return n_succ / n, n_switch / n, mean_sw, n_adapt / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/smerl_lowalpha_ckpt")
    ap.add_argument("--ckpt-file", type=str, default="ckpt_step010000.pt")
    ap.add_argument("--value-net", type=str, default="value_net_skill.pt")
    ap.add_argument("--skills", type=str, default="1,2")
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--force-start", type=int, default=2,
                    help="skill all models are forced to commit to first (wall on its path)")
    ap.add_argument("--max-steps", type=int, default=100)
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
    n_skills = cfg["n_skills"]
    sr = cfg.get("success_radius", 0.25)
    allowed = [int(s) for s in args.skills.split(",")]
    value_net, _ = load_value_net(os.path.join(run_dir, args.value_net), device)
    value_net, _, _ = wrap_value_norm(value_net, allowed, agent, cfg, start, 0.0,
                                      device, n_skills)

    # one fixed wall set, shared by every model (depends only on agent+skills)
    z_block, z_spare = args.force_start, [s for s in allowed if s != args.force_start][0]

    def env_fn(wall):
        return WallPoint2DGoalEnv(start=tuple(start), goal=tuple(goal), wall=wall,
                                  success_radius=sr, max_episode_steps=args.max_steps)

    walls = valid_walls(agent, env_fn, z_block, z_spare, np.random.default_rng(args.seed),
                        cap=args.n_trials)
    print(f"[ablate] {len(walls)} shared walls (block {z_block}, spare {z_spare}); "
          f"all models forced to start on skill {z_block}\n")

    # offline dataset for the switch analytic
    store = [os.path.join(args.run, "trajectories_dart12"),
             os.path.join(args.run, "trajectories_dagger12")]
    store = [os.path.join("src/smerl", s) for s in store]
    ds = SeqDataset(store, SequenceSpec(
        pattern=[["state", "skill"], ["skill", "action"], "value"], decouple=True))
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)

    hdr = (f"{'ablation':38s} | {'OFFLINE switch':>22s} | "
           f"{'CLOSED-LOOP wall (forced start)':>40s}")
    print(hdr); print("-" * len(hdr))
    print(f"{'':38s} | {'acc':>6s} {'P(new)':>7s} {'P(stay)':>7s} | "
          f"{'goal':>6s} {'switch':>7s} {'sw@step':>8s} {'ADAPT':>7s}")
    print("-" * len(hdr))
    for label, ckpt, dec, sw, lg in RUNS:
        path = os.path.join(run_dir, ckpt)
        if not os.path.exists(path):
            print(f"{label:38s} |  (missing {ckpt})")
            continue
        model = load_bc(path, device)
        sm = switch_metrics(model, loader, device)
        gr, swr, msw, adapt = closed_loop_wall(model, agent, value_net, cfg, start,
                                               goal, sr, n_skills, walls,
                                               args.force_start, args.max_steps, device)
        print(f"{label:38s} | {sm.get('acc',0):6.3f} {sm.get('p_new',0):7.3f} "
              f"{sm.get('p_stay',0):7.3f} | {gr:6.3f} {swr:7.3f} {msw:8.1f} {adapt:7.3f}")
    print("-" * len(hdr))
    print("\nOFFLINE acc = skill-head argmax==new at the labeled switch-decision steps "
          "(teacher-forced; want high).")
    print("CLOSED-LOOP (all forced to start on the stuck skill, same walls):")
    print("  goal    = reached the goal")
    print("  switch  = switched at all (gameable: a late or step-1 flip both count)")
    print("  sw@step = mean step of the FIRST switch (stall-driven ~20+, default-away ~1)")
    print("  ADAPT   = committed to the stuck skill, stalled, THEN switched to the escape "
          "and reached goal (first switch >= step 8). The honest closed-loop number.")


if __name__ == "__main__":
    main()
