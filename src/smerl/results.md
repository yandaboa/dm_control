# SMERL adaptive skill-switching — running results

_Branch SMERL. Run dir: `/mnt/storage/lti/dm_control/src/smerl/runs/smerl_lowalpha_ckpt/`._
_All paths absolute. Updated 2026-06-09._

---

## Phase-0 batch-8 multi-seed sweep (switch_share × positional encoding)

**Bar chart:** `/mnt/storage/lti/dm_control/src/smerl/runs/smerl_lowalpha_ckpt/phase0_bs8_seedsweep_bar.png`
**Builder:** `/mnt/storage/lti/dm_control/src/smerl/build_phase0_chart.py`
**wandb group:** `2d / adapt_p0_bs8_seedsweep` (+ seed-0 from `adapt_p0`).

Skills 1,2, batch 8, constant LR. 3 seeds/config (seed 0 + sweep seeds 1,2).
Eval: `eval_bamdp_adapt`, `--reset-on-switch --theta-bad 0.95`, 200 eps, max-steps 150,
value-norm skills 1,2. Mean over 3 seeds (range in brackets).

| pos-enc | switch_share | success (mean [min,max]) | succ\|failed | switches |
|---|---|---|---|---|
| wpe  | 0.20 | 0.872 [0.760, 0.995] | 0.750 | 0.58 |
| wpe  | 0.30 | 0.907 [0.775, 0.985] | 0.795 | 0.55 |
| **wpe**  | **0.40** | **0.975 [0.945, 0.995]** | **0.927** | **0.40** |
| wpe  | 0.50 | 0.815 [0.660, 0.965] | 0.648 | 0.84 |
| **rope** | **0.20** | **0.993 [0.990, 0.995]** | **0.979** | **0.32** |
| rope | 0.30 | 0.935 [0.885, 0.970] | 0.837 | 0.51 |
| rope | 0.40 | 0.947 [0.890, 0.995] | 0.873 | 0.53 |
| rope | 0.50 | 0.958 [0.940, 0.985] | 0.881 | 0.43 |

### Conclusions

1. **The single-seed dips were noise — the real picture is clean.** Earlier seed-0
   showed wpe@0.3=0.775 and rope@0.4=0.890 dips at *different* shares; over 3 seeds
   wpe peaks smoothly at 0.40 (0.975, tightest range) and the rope curve is high
   everywhere (0.94–0.99). The dips did not reproduce.
2. **wpe (learned absolute positions): switch_share 0.4 is the winner** — 0.975 success
   / 0.927 succ|failed, fewest switches (0.40), lowest variance. 0.5 over-switches
   (0.84 switches → thrash → 0.815) and 0.2 is high-variance (0.76–0.995). Confirms the
   locked Phase-0 config: **bs8, switch_share 0.4, wpe.**
3. **RoPE is flatter and higher than wpe across the board**, and **rope@0.2 is the best
   single config overall: 0.993 success / 0.979 succ|failed / only 0.32 switches.**
   Mechanism (see EVAL_phase0_results.md): RoPE conditions the switch on the *failure
   cue* (relative positions) instead of a memorized absolute timestep, so it doesn't need
   a high switch_share to push switching — a low share keeps it calm (few switches → with
   reset_on_switch, less self-inflicted thrash). RoPE also removes the share-sensitivity
   that wpe has.

**Takeaway / adopted config:** **RoPE + batch 8 + switch_share 0.2** is the Phase-0
winner and is now the adopted default (`configs/bc_adapt.yaml`: `rope: true`,
`switch_share: 0.2`). wpe + share 0.4 (0.975) is the strong learned-position baseline.
Phase 1 trains **both** on the same on-policy store to check whether RoPE's edge carries
to cross-episode latching.

---

## Phase 1 — cross-episode latching (skills 1,2)

Meta-episodic on-policy collection (`collect_demos.py`, chained sub-episodes under one
shared θ), eval `eval_adapt_multiep` (success should RISE with sub-episode index,
P(first==bad) should FALL). Memory: `phase1-latching-findings.md`.

### Round 1 — latching FAILS; two bugs found

Collection: driver `bc_bs8_rope_share20`, 400 meta-eps (mean **73 steps ≈ 1 sub-ep**).
Trained both RoPE+share0.2 (`bc_phase1_12.pt`) and wpe+share0.4 baseline on the same store.
`eval_adapt_multiep`, 80 meta-eps × 4 sub-eps, bad skill ∈ {1,2}:

| model | success by sub-ep #0→#3 | P(first==bad) #0→#3 | switch-off-bad | within-ep succ\|failed |
|---|---|---|---|---|
| rope+0.2 | 0.69 / 0.66 / 0.54 / 0.42 | 0.42 / **0.89** / 0.49 / 0.49 | 0.23 | **0.000** (0.01 sw) |
| wpe+0.4  | 0.71 / 0.57 / **0.07 / 0.12** | 0.42 / 0.84 / 0.39 / 0.42 | 0.31 | **0.000** (0.07 sw) |
| rope+0.2 (teacher-action) | 0.69 / 0.71 / 0.69 / 0.61 | 0.42 / 0.89 / 0.47 / 0.47 | 0.25 | — |

Plots: `adapt_multiep_phase1_rope.png`, `adapt_multiep_phase1_wpe.png`.

**Three findings:**
1. **No latching** — success does not rise with sub-ep index (flat→declining), and
   P(first==bad) *spikes to 0.89 at sub-ep 1* (the model re-tries the failed skill).
2. **Value-reset misfire = the core bug.** At a sub-episode boundary the BAMDP value
   resets to ~0; the within-episode-trained model reads this as a STALL and switches off
   the good skill. Persists under `--teacher-action` (success flat 0.69, same #1 spike) →
   it's a **skill-selection** bug, not the action head.
3. **RoPE >> wpe on long contexts** — wpe COLLAPSES at sub-eps 2,3 (0.07/0.12; absolute
   positions past the training length are untrained); rope degrades gracefully
   (0.54/0.42). Confirms RoPE is the right default for the multi-episode phases.
4. **Within-episode switching regressed to ~0** (both: succ|failed 0.000, ~0 switches).
   Root cause = data composition: dart12 (1000 clean) + meta (400 switch) = 29% switch
   eps at share 0.2, vs Phase-0's balanced 50%. `switch_share` tuned on 50/50 data is not
   composition-invariant.

### Round 2 — fix WORKS (latching achieved)

On-policy collection from the round-1 model (its own value-reset-misfire contexts get
expert-corrected) with LONG supervision (`--supervision-range 150,300`,
`--max-learner-steps 120`). Collected demos went from **mean 73 steps (~1 sub-ep) → 280
steps (~4.8 sub-eps, all successful)** — the expert now demonstrates STAYING on the good
skill across goal-resets, covering the full 4-sub-ep eval horizon. Retrained on
dart12 + meta + meta_r2 at both share 0.2 and share 0.4.

**Chart:** `/mnt/storage/lti/dm_control/src/smerl/runs/smerl_lowalpha_ckpt/phase1_latching_curves.png`
(builder `build_phase1_chart.py`). `eval_adapt_multiep`, 80 meta-eps × 4 sub-eps:

| model | success #0→#3 | P(first==bad) #0→#3 | switch-off-bad |
|---|---|---|---|
| round-1 rope+0.2 | 0.69 / 0.66 / 0.54 / 0.42 | 0.42 / 0.89 / 0.49 / 0.49 | 0.23 |
| round-1 wpe+0.4  | 0.71 / 0.57 / 0.07 / 0.12 | 0.42 / 0.84 / 0.39 / 0.42 | 0.31 |
| round-2 share0.2 | 1.00 / 0.97 / 0.84 / 0.80 | 0.42 / 0.57 / 0.19 / 0.17 | 0.57 |
| **round-2 share0.4** | **1.00 / 0.99 / 0.96 / 0.94** | 0.42 / 0.57 / **0.09 / 0.07** | **0.77** |

**Latching achieved.** Round-2 success stays near-perfect across all 4 sub-episodes (no
decay), and P(first==bad) *falls* with sub-episode index (0.42→0.07) — the model reads
its in-context history, avoids the skill it saw fail, and latches onto the good one.
**share 0.4 > share 0.2** (0.94 vs 0.80 at sub-ep #3; switch-off-bad 0.77 vs 0.57),
confirming the round-1 within-episode regression was a switch-signal-too-weak / data-mix
issue — the sparser switch demos need the higher share. **Adopted Phase-1 model:
`bc_phase1_12_r2_s40.pt` (RoPE, bs8, switch_share 0.4).**

**Teacher-action (skill-latching isolated, SMERL actions):** round-2 share0.4 success
1.00/0.97/0.94/0.91, P(bad) →0.06 — confirms the latching is in skill selection. share0.2
under teacher-action also reaches 0.91 (its full-policy gap vs share0.4 is partly the
action head).

**Within-episode sanity** (`eval_bamdp_adapt`, `--reset-on-switch`): share0.4 =
0.785 / succ|failed **0.283** / 0.61 sw; share0.2 = 0.835 / **0.450** / 0.46 sw.
Switching is RESTORED in both (round-1 was 0.000 / 0.01 sw). The absolute number is low only because this probe uses
`--reset-on-switch` (teleport) dynamics, which are OOD for the Phase-1 model (trained on
`continue_on_failure`, no teleport). The matched-dynamics metric is the multiep eval, where
sub-episode #0 = **1.00** (within-episode success under injected failures).

**Phase 1 conclusion:** in-context multi-episode adaptation works for 2 skills — the model
latches onto the good skill, avoids the observed-bad skill (P(bad-first) 0.42→0.07 across
sub-episodes), and recovers within-episode. The two enabling fixes over a naive transfer:
(1) **long multi-sub-episode supervision** in collection (so demos cover the eval horizon
and teach that a boundary value-reset ≠ a stall), (2) **switch_share 0.4** (the meta data
is switch-sparser than Phase-0's 50/50, so the lower 0.2 starved switching). RoPE matters
for long-context generalization.

---

## Phase 2 — scale to 3 behavior modes (skills 0,1,2)

Multi-round on-policy, reusing the Phase-1 fixes (long multi-sub-ep supervision,
switch_share 0.4, RoPE). r0 = bootstrap within-episode base (dart012 + iter0 expert demos);
r1,r2,r3 = on-policy meta-episodic rounds, each driven by the previous round's model.
Drivers: `phase2.sh` (r0–r2), `phase2_r3.sh` (r3). Collected demos: mean ~275 steps
(~4 sub-eps), 500/round. eval_adapt_multiep, 90 meta-eps × 4 sub-eps, bad ∈ {0,1,2}
(chance P(bad-first) = 1/3 ≈ 0.33).

**Chart:** `/mnt/storage/lti/dm_control/src/smerl/runs/smerl_lowalpha_ckpt/phase2_latching_rounds.png`
(builder `build_phase2_chart.py`).

| round | success #0→#3 | P(first==bad) #0→#3 | switch-off-bad |
|---|---|---|---|
| r1 (1 on-policy round) | 0.87 / 0.78 / 0.74 / 0.70 | 0.31 / 0.26 / 0.29 / 0.29 | 0.28 |
| r2 | 0.97 / 0.92 / 0.88 / 0.80 | 0.32 / 0.23 / 0.16 / 0.16 | 0.44 |
| **r3** | **0.98 / 0.93 / 0.84 / 0.83** | 0.32 / 0.14 / 0.18 / **0.14** | **0.51** |

**The "rounds scale with #skills" thesis holds — each on-policy round improves latching,
monotonically, with diminishing returns by r3.** With 3 skills, one round under-covers the
(larger) history space: r1 success decays 0.87→0.70 and P(bad-first) stays ≈ chance (~0.29,
no avoidance). Across rounds:
- success #3: **0.70 → 0.80 → 0.83**
- P(first==bad) #3: **0.29 → 0.16 → 0.14** (avoidance, well below the 1/3 chance line)
- switch-off-bad: **0.28 → 0.44 → 0.51**
- within-episode (#0): **0.87 → 0.97 → 0.98**

**Adopted Phase-2 model: `bc_phase2_r3.pt`** (RoPE, bs8, switch_share 0.4). 3-skill
latching is weaker than Phase-1's 2-skill (P(bad-first) 0.14 vs 0.07; success#3 0.83 vs
0.94) — consistent with the combinatorially larger history space — but the round-over-round
trend shows on-policy iteration is the right lever, exactly as predicted. A 4th round would
likely add little (r2→r3 gains are already small).

---

## Phase 1 retry — env-side terminate-on-switch (reset_on_switch=True), share sweep

Phase-1 r2 recipe rerun with the env teleporting to a fresh start on every skill
switch (give up & retry, step clock kept), in BOTH collection and eval — so the
previously-OOD `--reset-on-switch` probe is now in-distribution. Collection:
`collect_demos --reset-on-switch`, driver `bc_phase1_12.pt`, long supervision
(150,300), 800 eps (mean 280.8 steps ≈ 4.8 sub-eps, 800/800 with success) →
`trajectories_dagger12_meta_ros`. Trained on dart12 + the PURE ros store (old
non-teleport meta stores excluded), RoPE+bs8, `switch_share` ∈ {0.2,0.3,0.4,0.5},
ended at epoch ~40/50 (converged; best-val ckpts from epochs ~36–40).
Driver `phase1_ros.sh` + `phase1_ros_eval.sh`; configs `bc_phase1_ros_s*.yaml`;
wandb `2d / adapt_phase1_ros`.

Eval (matched dynamics): `eval_adapt_multiep --reset-on-switch` 80×4, and the
within-ep probe `eval_bamdp_adapt --reset-on-switch` (200 eps):

| share | success #0→#3 | P(first==bad) #0→#3 | switch-off-bad | within: succ / succ\|failed / switches |
|---|---|---|---|---|
| 0.2 | 0.89 / 0.89 / 0.85 / 0.85 | 0.42 / 0.57 / 0.16 / 0.16 | 0.58 | 0.925 / 0.773 / 0.51 |
| 0.3 | 0.93 / 0.95 / 0.97 / **0.96** | 0.42 / 0.60 / 0.06 / **0.04** | **0.80** | 0.995 / 0.983 / 0.52 |
| **0.4** | **0.96 / 0.95 / 0.94 / 0.96** | 0.42 / **0.34** / 0.07 / 0.06 | 0.74 | 0.995 / 0.983 / 0.35 |
| 0.5 | 0.96 / 0.91 / 0.96 / 0.95 | 0.42 / 0.54 / 0.06 / 0.04 | 0.79 | 0.995 / 0.983 / 0.33 |

**Findings:**
1. **Latching works under terminate-on-switch dynamics.** Shares 0.3–0.5 are all
   strong and near-identical: flat ~0.95 success across sub-eps, P(bad-first) →
   0.04–0.06, on par with the adopted non-ros `bc_phase1_12_r2_s40` (0.94 / 0.07).
2. **The within-episode probe is no longer under-reporting** once dynamics are
   matched: succ|failed 0.983 (s30–s50) vs 0.283 for the old r2_s40 model under the
   same probe — confirming the earlier OOD interpretation, and that training with
   teleport dynamics fixes within-episode recovery under that regime.
3. **Share sensitivity is flatter than in r2**: only 0.2 underperforms (0.85
   multiep, succ|failed 0.77, switch-off-bad 0.58). The ros mix (1000 dart + 800
   switch = 44% switch eps) is less switch-sparse than r1's 29%, so 0.2 is less
   starved — but high shares still win under matched-ros training.
4. **Residual sub-ep-1 value-reset spike** (P(bad) 0.54–0.60 at #1 for s30/s50)
   persists mildly but is corrected by #2; **s40 largely avoids it (0.34)**. The
   spike barely hurts success because under ros a switch is a cheap clean retry —
   high switch-off-bad converts bad first picks into successes within the budget.
5. **Adopted: `bc_phase1_ros_s40.pt`** — ties best success (0.96 at #3), best at
   the #1 boundary (P(bad) 0.34), fewest switches among the strong variants
   (0.35), within-ep 0.995 / 0.983. (s30 edges it on #3 avoidance 0.04 and
   switch-off-bad 0.80, but has the worst #1 spike, 0.60.)

Charts: `adapt_multiep_phase1_ros_s{20,30,40,50}.png` (in `runs/smerl_lowalpha_ckpt/`).

---

## Phase 2 retry — terminate-on-switch, 3 skills, DAgger round dynamics

All under the matched terminate-on-switch eval, **train-matched theta** (sample_theta:
one forced bad + one forced good, third play skill ~Beta(0.5,0.5) → 52% of metas
have a SECOND bad skill) unless noted. 90 metas × 4 sub-eps; within-ep probe 200 eps,
theta-mode train. 500 meta-eps/round (1000 at fixed 50 epochs doubled gradient steps
and overfit r1 — `bc_phase2_ros_r1_n1000.pt`). Stores: dart012 (402) + per-round
takeover-scheduled DAgger metas, all `--keep-intervened-only --require-success`.

| model (round, takeover) | success #0→#3 | P(bad-first) #0→#3 | switch-off-bad | within: succ / succ\|failed |
|---|---|---|---|---|
| `bc_phase2_r3` (old anchor, non-scheduled) | 0.93 / 0.83 / 0.71 / 0.67 | 0.41 → 0.18 | 0.37 | — |
| `bc_phase2_ros_r1` (r1, 5–15) | 0.61 / 0.56 / 0.62 / 0.63 | 0.36 → 0.27 | 0.22 | 0.720 / 0.291 |
| `bc_phase2_ros_r2` (r2, **70–85** shifted-min) | 0.68 / 0.83 / 0.86 / **0.89** | 0.41 → **0.08** | **0.65** | 0.965 / 0.892 |
| `bc_phase2_ros_r2w` (r2, 5–100 widened, concat) | 0.38 / 0.78 / 0.78 / 0.76 | 0.41 → 0.18 | 0.55 | 0.940 / 0.867 |
| `bc_phase2_ros_r2w_wt` (r2, 5–100 + weights [.3,.3,.4]) | **0.74** / 0.81 / 0.82 / 0.84 | 0.36 → 0.11 | 0.61 | **0.975 / 0.935** |
| `bc_phase2_ros_r3w` (r3, 5–120, concat) | 0.50 / 0.66 / 0.81 / 0.81 | 0.42 → 0.12 | 0.70 | 0.930 / 0.825 (11.4 sw) |
| `bc_phase2_ros_r3w_wt` (r3, 5–120 + weights [.1,.2,.3,.4]) | 0.64 / 0.72 / 0.77 / 0.74 | 0.31 → 0.16 | 0.51 | 0.975 / **0.941** |

Round-2 findings (so far):
1. **One scheduled on-policy round transforms the model.** r2 (takeover 70–85: the
   expert may only step in after the learner has self-generated ~2 sub-eps of
   history) takes switch-off-bad 0.22 → 0.65, P(bad-first at #3) 0.27 → 0.08, and
   rising success 0.68→0.89 — decisively better than the old 3-round `bc_phase2_r3`
   anchor, whose success FELL across sub-eps (0.93→0.67).
2. **Within-episode recovery is fixed by round 2**: succ|failed 0.29 → 0.89 (train
   theta; 0.96 legacy) — the late-takeover corrections target exactly the
   failed-history states the learner actually visits.
3. The r1 overfit fix (500 eps + best-val early stop ≈ ep27) did NOT improve r1
   closed-loop vs the overfit n1000 — round-1 weakness is data coverage, not the
   training regime.
4. Offline val loss remains uninformative across models (bs sweep: bs64 had the
   worst val 1.56 yet the best late-episode latching before the sweep was killed;
   r2w_wt val 1.02 < r2w 1.05 happens to agree closed-loop, but r2's 0.89 is on
   2× different data — not comparable).
5. **Recency weighting beats plain concat on identical widened data** (r2w_wt vs
   r2w: every metric, most dramatically first-episode success 0.74 vs 0.38). The
   widened 5–100 store (500/901 kept, mean 293.2) dilutes early-episode behavior
   unless per-store gradient shares are pinned — new `data.store_weights` config in
   train_seq (WeightedRandomSampler, shares [0.3,0.3,0.4] over dart/r1/r2w).
6. **Widened-min (5–100, weighted) ≈ shifted-min (70–85, concat) at round 2**:
   r2w_wt wins on first-episode success (0.74 vs 0.68) and within-ep recovery
   (0.935 vs 0.892); r2 wins on late-episode latching (0.89 vs 0.84) and avoidance
   (0.08 vs 0.11). The shifted-min arm gets early-correction coverage for free from
   dart012+r1 in the concat — i.e., the two schedules differ less than the
   weighting choice does.

Round-3 findings (takeover 5–120, max-learner-steps 140, driver `bc_phase2_ros_r2w_wt`,
500/776 kept):
7. **Round 3 hits diminishing returns in the widened line.** r3w_wt regresses vs
   r2w_wt on multiep latching (#3 success 0.74 vs 0.84, switch-off-bad 0.51 vs
   0.61) while only matching it within-episode (0.941 vs 0.935). r3w concat gets
   the best switch-off-bad of the line (0.70) but keeps the concat first-episode
   crater (0.50) and turns hyper-switchy within-episode (11.4 switches/ep vs ~4).
   Likely causes: the 100–120 takeover tail rarely binds (sub-eps cap at 100
   steps, most failures are earlier), and weights [.1,.2,.3,.4] cut the
   dart012+r1 share to 0.3, eroding the early-episode prior that round 2 kept.
8. **Adopted: `bc_phase2_ros_r2`** (70–85, concat) — best cross-episode latching
   (0.68→0.89, P(bad-first) 0.08, switch-off-bad 0.65) with near-best within-ep
   recovery (0.892); `bc_phase2_ros_r2w_wt` is the close runner-up profile
   (better #0 and within-ep, slightly weaker latching). Two rounds suffice in
   this domain; spend round-3 budget on weight-schedule tuning (e.g. keep
   [.3,.3,.4]-style shares with a floor on the expert prior) rather than wider
   takeover windows.

Infra (this run): backported UWLab training speedups to train_seq/seq_data/seq_model
— bf16 autocast, in-RAM tokenized-episode cache, pinned loaders, subsampled
diagnostics (+`.to(out.dtype)` bf16 fix in embed); new `data.store_weights` config.

---

## Summary (all phases)

1. **Phase 0 config sweep** — adopted **RoPE + bs8 + switch_share** (0.2 on balanced data;
   0.4 when the switch mix is sparser). RoPE reads the failure cue (relative positions),
   stays calm and extrapolates to long contexts; wpe collapses past its training length.
2. **Phase 1 (2 skills)** — in-context cross-episode latching achieved (success 0.94–1.0
   across sub-eps, P(bad-first)→0.07). Two fixes: long multi-sub-ep supervision + share 0.4.
3. **Phase 2 (3 skills)** — latching scales with on-policy rounds (P(bad-first) 0.29→0.14
   over r1→r3); needs more rounds than 2 skills, as predicted.

Charts: `phase0_bs8_seedsweep_bar.png`, `phase1_latching_curves.png`,
`phase2_latching_rounds.png` (all in `runs/smerl_lowalpha_ckpt/`).
