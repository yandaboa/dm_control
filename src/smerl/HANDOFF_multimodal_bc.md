# Handoff — Multimodal skill-token BC transformer (SMERL 2D point-goal)

Date: 2026-06-03. Branch `SMERL`. Nothing committed this session — all files below
are new/modified in the working tree.

## Goal

Train ONE transformer that behavior-clones all SMERL skills such that, at a
*multimodal* state, it can **sample a skill and commit to executing that skill's
actions**. Built on the preferred policy `smerl_lowalpha_ckpt/ckpt_step010000.pt`
(5 skills, start≈[-0.25,-0.5], goal≈[0.66,0.66], success_radius 0.25) and the
skill-conditioned value net `value_net_skill.pt`.

## Core design — token layout `[s_t, z_t, v_t]` per timestep

Sequence fed to the GPT2 backbone: `[s_0, z_0, v_0, s_1, z_1, v_1, ...]`.

- `s_t` (state): input, no target.
- `z_t` (skill, **discrete** — nn.Embedding encoder + **categorical** head):
  predicted from the **state** token's hidden (CE loss). Skill tokens ARE fed
  back into context.
- `a_t` (action, **gaussian** head + NLL): predicted from the **skill** token's
  hidden. Action tokens are NOT fed back into context.
- `v_t = V(s_t, z_t)` (skill-conditioned value): context-only input, no target.

**Why z BEFORE v (the key decision):** the value is skill-conditioned, so the
original `[s,v,z]` order was circular at rollout (need z to compute v, but z is
sampled from v). Putting z first — chosen from the state alone — breaks the cycle:
at rollout we sample z_t from s_t, commit, THEN look up V(s_t, z_t). Needed ZERO
change to data collection: stored `values` are already `V(s_t, z_demo) = v_t`.

### Value-function decision trail
- Rejected **task-oriented V(s)** and plain **skill-conditioned V(s,z)** in favor
  of trying **max-of-ensemble** `V_max(s)=max_z V(s,z)` (skill-agnostic but built
  from the preferred ensemble).
- **`V_max` FAILED the gate check** (`eval_value_ensemble.py`): argmax-match
  `argmax_z' V(s,z')==z_true` = 0.154 (below chance 0.20); collapses to "skill 3's
  value" because in a 2D nav env every skill can reach the goal from anywhere, so
  the max tracks the globally-best skill. (Memory: `value-ensemble-max-fails`.)
- Resolved by the `[s,z,v]` reorder + keeping skill-conditioned `V(s,z)`.

## Files

### New
- `configs/bc_multimodal.yaml` — the config (pattern `[[state,skill],[skill,action],value]`,
  heads `{skill: categorical, action: gaussian}`, n_embd 64 / n_layer 6 / n_head 4,
  dropout 0.1, 50 epochs, lr 3e-4, wd 1e-2, batch 64).
- `eval_value_ensemble.py` — the V_max gate check (argmax-match + |Vmax-Vz|).
- `eval_bc_multimodal.py` — closed-loop eval. `rollout_multimodal(force_z=...)`:
  if `force_z` is None, **sample** z_t ~ p(.|s_t); else **clamp** z_t. Reports
  per-skill forced success vs teacher, and sampled commitment/multimodality.
- `eval_perskill_fit.py` — expressiveness vs covariate-shift diagnostic
  (on-distribution mean-fit MSE vs closed-loop drift / rollMSE).
- `plot_skill_replication.py` — (1) `--select` picks the best checkpoint by
  closed-loop **forced-skill success** and copies it to `bc_multimodal.pt`;
  (2) plots teacher (grey) vs transformer forced-z rollouts (color) per skill.

### Modified
- `seq_model.py` — `TrajectoryGPT` now takes `discrete={name:n_classes}`
  (embedding encoder + categorical head) and `head_types={name: mse|gaussian|
  categorical}`. Added `_loss_for` (shared loss helper), `modality_loss`
  (differentiable per-term loss from precomputed hidden — for grad diagnostics),
  `head_logits`, `target_loss_per_position`. `embed`/`forward` handle discrete
  modalities.
- `seq_data.py` — added `skill` modality (dim 1) sourced from `ep["z"]` via
  `_slot_vec`; `make_modalities` includes it.
- `train_seq.py` — wires `discrete`/`head_types` from config and persists them in
  the checkpoint blob; logs **per-skill action loss** (`act_z0..z4`), **per-term
  gradient norms** (`gradnorm_skill/action`, `_bb` backbone-only, `grad_cos_bb`),
  and `--save-every N` periodic raw checkpoints (for closed-loop selection).

### Data
- Store `runs/smerl_lowalpha_ckpt/trajectories_multimodal` — 500 clean demos
  (100/skill, `--p 0` = no forced failures, honest `V(s,z)`, max-steps 75).
  Collected via existing `collect_trajectories.py` (all skills, value_net_skill).

## Findings

1. **Multimodality + commitment WORK.** Sampling z from the shared start:
   committed-skill histogram `[48,38,32,35,47]` over 200 eps (all 5 skills),
   **within-episode commitment 0.993**.
2. **Not a capacity problem.** Per-skill open-loop action NLL is uniform
   (0.79/0.79/0.95/1.12/1.32 for z0..4) and its ordering is *inverted* vs
   closed-loop success → expressiveness is fine. `eval_perskill_fit`: on-fit MSE
   uniform (0.08–0.11), but closed-loop rollMSE 4–9× higher and drift-to-demo-
   manifold 10–30× the within-demo spacing → **covariate shift / compounding error**.
3. **No multi-task conflict.** Gradient diag: skill CE dominates the backbone
   gradient early (ep0: skill-bb 4.5 vs action-bb 0.9), action NLL dominates after
   ep~9 (skill collapses to ~0.03 CE). **`grad_cos_bb` stays slightly POSITIVE
   (~0.02–0.09), never negative** → skill and action objectives don't fight over
   the trunk.
4. **Closed-loop success is unstable across epochs at smoothly-decreasing val
   loss.** Forced-skill success per checkpoint swings wildly: ep045 `[1,1,1,1,0]`
   (feas-mean 1.00) → ep048 `[0,.02,0,.7,0]` (0.18) → ep049 `[1,1,1,1,0]` (1.00).
   **Picking by val loss is unreliable** — must select by closed-loop. (Consistent
   with the standing finding in memory `bc-transformer-hparams`.)

## Current state

- **Best model selected: epoch 045**, `feas-mean=1.000` — forced-skill success
  `[1.0, 1.0, 1.0, 1.0, 0.0]`. Skills 0–3 clone the teacher perfectly; **skill 4 =
  0 because the TEACHER also fails skill 4 under T≤75** (needs ~80 steps), not a
  clone failure. Saved as `runs/smerl_lowalpha_ckpt/bc_multimodal.pt`.
- Plot `runs/smerl_lowalpha_ckpt/bc_skill_replication.png` — transformer forced-z
  rollouts (color) track the teacher (grey) tightly for z0–3.
- So the earlier "skill 2 collapses / skill 1 partial" was a **checkpoint lottery**,
  not a per-skill learnability problem — with closed-loop selection all 4 feasible
  skills clone at 1.00.

## How to run

```bash
# collect (already done; store exists)
conda run -n SMERL python -m src.smerl.collect_trajectories \
  --run runs/smerl_lowalpha_ckpt --ckpt-file ckpt_step010000.pt \
  --n-per-skill 100 --p 0 --max-steps 75 \
  --out src/smerl/runs/smerl_lowalpha_ckpt/trajectories_multimodal

# train one model on GPU 0 (logs grad-norms + per-skill loss; saves ckpts)
CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API_KEY conda run -n SMERL python \
  -m src.smerl.train_seq --config src/smerl/configs/bc_multimodal.yaml \
  --run-name bc_multimodal --wandb-project 2d --wandb-group multimodal --save-every 3

# select best checkpoint by closed-loop forced-skill success + replication plot
CUDA_VISIBLE_DEVICES=0 conda run -n SMERL python -m src.smerl.plot_skill_replication \
  --run runs/smerl_lowalpha_ckpt --select --models "bc_multimodal_ckpts/ep*.pt"

# full multimodal eval (forced per-skill + sampled commitment)
CUDA_VISIBLE_DEVICES=0 conda run -n SMERL python -m src.smerl.eval_bc_multimodal \
  --run runs/smerl_lowalpha_ckpt --model bc_multimodal.pt --n-episodes 200
```
wandb project `2d`, group `multimodal`. Python prints are buffered — read logs
after the background task-notification fires, don't poll.

## Open threads / possible next steps

- **Closed-loop instability is the main open problem.** Root cause is covariate
  shift, not capacity/conflict. Candidate fixes: DAgger / on-policy correction,
  deterministic demos (cleaner action targets), action-output smoothing, more
  demos, or simply standardize on closed-loop checkpoint selection
  (`plot_skill_replication --select`) as the model-picking method.
- **Skill 4** needs horizon ≥ ~80 to be feasible at all (teacher fails @75).
- **BAMDP failure story not yet exercised here.** The value token is currently
  clean (`--p 0`). The `continue_on_failure` / value plateau-decline infra
  (`bamdp_env.py`) exists; a next phase could inject failures so `v_t` plateaus
  and test whether the model *switches* committed skill in response.
- Relevant memory: `multimodal-skill-token-bc`, `value-ensemble-max-fails`,
  `bc-transformer-hparams`, `preferred-smerl-policy`, `value-conditioning-ablation`,
  `bamdp-value-obs-infra`.
