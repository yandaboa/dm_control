# Adaptive Skill-Switching via In-Context BAMDP Adaptation — Plan & Understanding

_Last updated: 2026-06-09. Branch: SMERL._

## 1. The point of the project (what success means)

We train **one sequence model** (GPT over `[state, skill, value]` tokens) to act as an
**adaptive meta-policy** over a BAMDP whose hidden latent `theta` is the per-skill
failure rate. The model is given a frozen library of SMERL skills (a base policy +
skill-conditioned value net `V(s,z)`). Its job is to decide *which skill to commit to*,
and to **revise that decision using the success/failure information it accumulates in
context**.

The core capability is **in-context inference over multiple episodes**, not a fixed
reactive policy:

- **Latch onto success**: if a skill is seen to succeed earlier in context, keep
  picking it (from step 0 of later sub-episodes) instead of re-exploring.
- **Avoid observed failure**: if a skill's value stalls / it fails, stop committing
  to it.
- **Within-episode switch**: when the value-feedback tokens show the current skill
  has silently failed (value plateaus/declines), abandon it mid-episode and switch
  to one that works.

A meta-episode chains several sub-episodes under a **shared `theta`** with a persistent
transformer context (no context reset between sub-episodes), so the latent is constant
and the only way to do better over time is to *read the history*.

## 2. Why a learner MUST be inside DAgger (corrected reasoning)

This is **not** classic DAgger distribution-shift correction. The learner does not
"drift OOD" and we are not iterating to patch a policy that wanders off-manifold.

The reason on-policy collection is essential: **the context the model must learn to
interpret is the context the model itself generates** — which skills it tries, in what
order, and the success/failure signals those attempts produce. The latent `theta` is
inferred from *that* history. If the history is produced by a random / model-less
"learner" (as in the broken `run_3skill_experiment.sh`), the expert's latching
demonstrations are conditioned on contexts the trained model never actually visits, so
the learned latching is off-distribution and weak.

So: **the learner picks skills on-policy; the expert demonstrates the correct adaptive
behavior (switch off a stalled/bad skill, latch onto a good one) on the learner's own
self-generated histories.**

## 3. Known-good baseline (the "signs of life", 2 behavior modes)

Recovered from `trajectories_dagger12` manifest + `dagger_collect.py`:

- Skills **1,2** (two behavior modes).
- Base model **`bc_base_cat12.pt`** — a transformer pretrained on **DART** demos, with a
  categorical skill head — used on-policy to pick the attempted skill `z0`.
- **Simplified intervention trigger**: after the BAMDP `failing` flag turns on, wait
  `K ~ U[5,10]` steps, then the expert switches to a good skill `z_good`; teacher drives
  to the goal. Keep the episode iff it `intervened AND succeeded`.
- This demonstrated robust **within-episode** switch-on-stall.

## 4. Non-negotiable training tricks (always on)

1. **Decoupled skill target** — the skill *target* flips to `z_good` a few value tokens
   **after** the stall begins (not at failure onset; `obs_delay`/`decouple_delay` ~2–3),
   so the model is trained to switch from the *value evidence*, while the value is still
   visibly stalling.
2. **Switch reweighting (data-driven, one share knob)** — under decoupling the supervised
   target leads the executed switch, so the "switch" is a **decision window** of multiple
   tokens (`skill_target[t] != skills[t]`: target says switch while the old skill is still
   executing), not one step. Those decision tokens are scaled so they contribute a target
   fraction `switch_share` of the skill gradient: `w = (s/(1-s)) * n_nondecision /
   n_decision` per episode, spread across the whole window. Replaces the old raw
   `switch_weight=20`. **Phase-0 sweep: `switch_share=0.4` is best** and beats the old
   winner (0.5 over-switches and thrashes; 0.2 under-adapts). See EVAL_phase0_results.md.
   (`seq_data.py::SequenceSpec.tokens`, `balance_switch: true`, `switch_share: 0.4`.)
3. **Logic gate** — auxiliary BCE switch probe, supervised on **every** decision-window
   token (not just the transition); `pos_weight` is data-driven (`n_neg / n_pos`).

Configs encode these: `configs/bc_adapt.yaml`, `configs/bc_adapt_3skill.yaml`
(`decouple: true`, `balance_switch: true`, `logic: true`).

## 5. Plan

### Phase 0 — reproduce the 2-skill signs of life, on-policy (sanity)
Skills **1,2**. Canonical flow, learner in the loop:
1. `collect_trajectories` → DART base demos (skills 1,2).
2. `train_seq` → base transformer `bc_base_cat12.pt` (with the tricks; single fixed arch
   so it can be reused as the iteration driver).
3. `dagger_collect --model bc_base_cat12.pt --skills 1,2` → **on-policy** `z0`, simple
   `K~U[5,10]` trigger, expert→`z_good`, keep intervened&success.
4. `train_seq` on DART + DAgger → `bc_dagger12.pt`.
5. `eval_bamdp_adapt` → confirm within-episode **success-on-failed-episodes** and switch
   rate jump from base → dagger.

### Phase 1 — cross-episode latching (the real goal), skills 1,2
Move to **meta-episodic** on-policy collection so the model learns to read multi-episode
history. Fix the model-less bug: drive `collect_demos.py` with `--model <latest>` so the
learner picks skills on-policy across chained sub-episodes under a shared `theta`. Train
with the same tricks; evaluate with `eval_adapt_multiep.py`.

### Phase 2 — scale to 3 behavior modes, skills 0,1,2
Same on-policy meta-episodic recipe, `--skills 0,1,2`.

## 6. Metrics

- **Within-episode** (`eval_bamdp_adapt`): overall success, **success on episodes where a
  failure was actually injected**, mean #switches.
- **Cross-episode latching** (`eval_adapt_multiep`):
  - success rate by sub-episode index — should **rise** with index;
  - `P(first skill == bad)` by sub-episode index — should **fall** with index (avoidance);
  - switch-off-bad rate when committed to the bad skill.
- **Training-time (wandb, `switch/`)**: `acc`, `p_new`, `p_stay` over the full decision
  window; and `pnew_pos{k}` / `gate_pos{k}` — skill-head P(switch-to-good) and logic-gate
  switch probability by within-window position `k`. Want these to **rise with `k`**: the
  switch probability should grow as more failure-feedback tokens accrue in context.

## 7. On "rounds" (iteration) — reframed

Iterating the collect→retrain loop is **not** a drift fix. What actually justifies more
rounds here is **coverage of the in-context history space**, which scales with the number
of skills:

- What the model must learn to read is the *space of in-context histories* — which skill
  turned out bad, which good, and in what order they were tried / seen to succeed. With
  only **2 skills** that space is tiny (one bad, one good), so a **single on-policy round
  can plausibly cover all the histories needed**.
- As skills are added (3+), the number of distinct (bad-skill, tried-order,
  observed-success) histories grows combinatorially, and one round of on-policy
  collection only visits a fraction of them — so **more rounds become necessary** to
  cover the histories the model must adapt over.
- Secondary benefit: a model better at latching generates *cleaner* on-policy histories
  (locks onto the good skill sooner), so later rounds' expert demonstrations sit on
  better-formed contexts.

So: **Phase 1 (2 skills)** — default to a **single** on-policy round; add one only if
`eval_adapt_multiep` latching is still partial. **Phase 2 (3 skills)** — expect to need
**multiple** rounds to cover the larger history space.

## 8. Open questions / to confirm with Yanda

- Phase 1/2 use the meta-episodic `collect_demos.py` (with the learner wired in), not the
  single-episode `dagger_collect.py` — confirm.
- Whether to iterate at all beyond one on-policy round (see §7).
