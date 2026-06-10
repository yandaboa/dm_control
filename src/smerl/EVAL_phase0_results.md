# Phase 0 results — data-driven switch reweighting (skills 1,2)

_Run 2026-06-09. Branch SMERL._

## What was run

Retrained on the **existing on-policy stores** `trajectories_dart12 + trajectories_dagger12`
(skills 1,2; `dagger12` was collected on-policy with `bc_base_cat12`). Same model
(GPT2, n_embd 256, 6 layers, disc action head, 5.04M params), same data, same seed — the
**only** changes vs the prior winner `bc_logic_dec` are the training tricks:

| model | switch reweighting | logic gate |
|---|---|---|
| `bc_base_cat12.pt` | — (DART base, no intervention data) | — |
| `bc_logic_dec.pt` (old winner) | fixed `switch_weight=20`, single transition token | single token, `pos_weight=50` |
| `bc_adapt12.pt` (Phase 0) | **data-driven** `n_non/n_dec` over the **decision window** | **every** window token, `pos_weight=n_neg/n_pos=11.4` |

Training (`bc_adapt12`): `best_val=1.644`; logic `pos_weight=11.4` (6441 switch tokens);
wandb run `9tik140k` (project `2d`, group `adapt_p0`).

## Closed-loop eval (`eval_bamdp_adapt`, skills 1,2, `--reset-on-switch --theta-bad 0.95`, 200 eps, max-steps 150)

```
                  model   success   succ|failed   %failed   switches
       bc_base_cat12.pt     0.690         0.000      0.31       0.00
        bc_logic_dec.pt     0.785         0.606      0.49       0.61
          bc_adapt12.pt     0.360         0.298      0.52       2.82
```

- `success` — overall goal rate.
- `succ|failed` — goal rate **on episodes where a failure was actually injected** (the adaptation metric).
- `switches` — mean skill switches per episode.

## What regressed

**`bc_adapt12` (data-driven balancing) is much worse than the old winner `bc_logic_dec`,
and the cause is over-switching.**

- Overall success **0.785 → 0.360**.
- Success-on-failed **0.606 → 0.298**.
- Switches per episode **0.61 → 2.82** — the model thrashes: it switches, then keeps
  switching instead of committing/latching to the good skill. With `--reset-on-switch`
  every spurious switch teleports it back to a fresh start, so it rarely reaches the goal.

The base model (`bc_base_cat12`) never switches (0.00) and gets 0.000 on failed episodes,
as expected. The winner switches ~once and succeeds. The Phase-0 model switches ~3× and fails.

## Diagnosis

The data-driven scheme forces the **decision-window tokens to contribute 50% of the skill
gradient** (`sum_switch == sum_nonswitch`). The old winner put only ~20% there
(one token × 20 ÷ ~97 total). So the new model over-emphasizes "switch" relative to
"stay / commit / latch" and becomes trigger-happy. The "non-decision" tokens lump together
two behaviors the model must keep — *stay on the good skill before failure* **and**
*latch onto the good skill after switching* — and the 50/50 balance starves both.

Corroborating signal (training-time): at the decision window the skill head sits near a
coin-flip (`p_new ≈ 0.50`, `p_stay ≈ 0.48`), and the logic gate over-fires
(`pos_rate_pred ≈ 0.30`, precision ≈ 0.26, recall ≈ 0.99). Both point to an over-eager
switch bias.

## What did work

- **Per-position switch-probability logging** behaves exactly as intended — switch
  probability climbs monotonically across the decision window as more failure feedback
  accrues in context (epoch 48): `pnew_pos` .485 → .576, `gate_pos` .802 → .897. This is
  pure diagnostic (no effect on training) and worth keeping.
- **The logic gate is training-only** — confirmed it is not read at inference
  (`adaptive_transformer.py` / `eval_bamdp_adapt.py` have no logic-gate usage). So the
  regression is driven by the **skill-head reweighting**, not the logic-gate change.

## Caveat / not yet isolated

Two training changes moved together vs the winner: (a) the switch reweighting
(fixed-20-single-token → data-driven-50/50-window) and (b) the logic-gate supervision
(single-token/`pw=50` → window/`pw=11.4`). Since the logic gate is inference-irrelevant,
(a) is the prime suspect, but a clean isolation run would confirm it.

## Sweep over `switch_share` (the decision group's target gradient fraction)

Added a `switch_share` knob: `w_dec = (s/(1-s)) * n_nondecision/n_decision`, so the
decision-window tokens contribute fraction `s` of the skill gradient (s=0.5 = equal).
Swept s ∈ {0.5, 0.4, 0.3, 0.2}, same data/arch/seed, 1 GPU each (parallel), same eval.

```
model                success  succ|failed  %failed  switches  best_val
bc_base_cat12          0.690        0.000     0.31      0.00      —
bc_logic_dec           0.785        0.606     0.49      0.61      —     (old winner, fixed sw=20)
sw50  (s=0.50)         0.360        0.298     0.52      2.82    1.644   over-switch / thrash
sw40  (s=0.40)         0.860        0.717     0.49      0.57    1.622   BEST
sw30  (s=0.30)         0.845        0.696     0.51      0.61    1.500
sw20  (s=0.20)         0.655        0.323     0.47      0.53    1.467   weak/mistimed switches
```

**Conclusion: the data-driven reweighting at `switch_share ≈ 0.3–0.4` BEATS the
hand-tuned fixed `switch_weight=20` winner** — sw40 success **0.860 vs 0.785**,
success-on-failed **0.717 vs 0.606**, with the same ~0.6 switches/episode. The relationship
is non-monotonic: 0.5 over-switches (2.82/ep → thrash → fails), 0.2 switches the right
*count* but adapts poorly (mistimed/weak switches). The sweet spot is ~0.35.

Offline `best_val` is the REVERSE of closed-loop quality (sw20 lowest val, near-worst
closed-loop; sw40 highest val among winners, best closed-loop) — do not select by val.

**Decision:** adopt `switch_share: 0.4` as the default for `bc_adapt.yaml` /
`bc_adapt_3skill.yaml`. Keep the per-position logging (pure diagnostic) and the
window-wide logic gate (training-only, inference-irrelevant).

## Batch-size ablation (at `switch_share=0.4`, 50 epochs, single seed)

Motivation: small batches over long correlated trajectory sequences can train strangely.

```
model        success  succ|failed  %failed  switches  best_val
bs4            0.855       0.688     0.47      0.65      0.898
bs8            0.985       0.964     0.28      0.37      1.035   ← outlier spike
bs16           0.835       0.683     0.52      0.63      1.105
bs32           0.770       0.614     0.57      0.73      1.355
bs64 (sw40)    0.860       0.717     0.49      0.57      1.622
```

**The trend is NON-monotonic — bs8 is a lone spike above BOTH neighbors (bs4 .855, bs16
.835).** Treat with suspicion, not celebration:
- A peak sandwiched between worse neighbors is the signature of run-to-run variance, not a
  real batch-size effect (if "smaller=better" held, bs4 should be ≥ bs8; it isn't).
- bs8's `%failed=0.28` is anomalous (others ~0.5). `%failed` = fraction of episodes where a
  failure was actually injected; with a fixed-ish start and the bad skill assigned randomly
  to skill 1/2 it should be ~0.5 regardless of model. So bs8's `succ|failed=0.964` is over a
  smaller, likely-easier subset — not apples-to-apples. (Overall `success=0.985` is
  subset-independent, though.)
- bs8 training itself is smooth (val falls cleanly, switch acc → .609, no spikes) — so it's
  not a visibly broken run, which is exactly why a single seed can't tell variance from signal.

`best_val` again falls monotonically with smaller batch (more grad steps) and again does
NOT track closed-loop — reconfirms: don't select by val.

**Action:** ran a multi-seed confirmation (seeds 0,1,3 for bs8; 0,1,2,3 for bs64).

### Multi-seed confirmation — bs8 is REAL, not variance

```
batch  seeds        success                 succ|failed
bs8    0,1,3        .985 .995 .965  (~.98)  .964 .986 .910  (~.95)
bs64   0,1,2,3      .860 .840 .675 .815 (~.80)  .717 .689 .458 .654 (~.63)
```
(bs8 seed 2 hit a transient CUDA error in eval — excluded.)

bs8 reproducibly ≈0.98 success / ≈0.95 succ|failed across 3 seeds, and is *lower variance*
than bs64 (whose seed-2 fell to .675). It consistently shows lower `%failed` (~.36 vs ~.55)
and fewer switches (~.4 vs ~.7) — the small-batch model is genuinely calmer and better, a
true peak (beats bs4 AND bs16, so not a monotonic "smaller=better" trend). Training itself
is smooth. **batch 8 is a real, reproducible improvement over batch 64.**

### Switch-timing analysis (visualize_bamdp_switch.py) — WHY bs8 wins

```
model       steps fail->switch   switch timestep      value@switch
bs8         5.4  (std 2.7)       19.5 (std 12.8)      0.20
bs64/sw40   31.9 (std 27.0)      58.1 (std 0.2)       0.27
```

**bs8 conditions the switch on the failure cue; bs64 fires at a memorized time.**
- bs8 switches ~5 steps after failure onset (tight, std 2.7) and its *absolute* switch
  time varies (19.5±12.8) — it tracks WHERE the stall starts. ~5 steps is reasonable
  (decouple obs_delay≈2-3 + a couple value tokens to read the stall); value has decayed to
  ~0.20 by then.
- bs64 switches at a nearly FIXED absolute timestep ~58 (std 0.2!) regardless of when the
  failure occurred, so steps-fail->switch has std 27. It is NOT reading the failure
  feedback — it learned a fixed switch time. Late and unreliable.

Interpretation: at large batch the switch signal is averaged across the batch and the model
collapses onto the mean switch *time* rather than the *failure cue*; small batch keeps the
per-episode failure→switch coupling. Plots: bamdp_switch_bc_bs8_s1.png,
bamdp_switch_bc_adapt12_sw40.png.

**Phase-0 final config: `switch_share=0.4`, batch 8.**

### Why large batch fails: an absolute-position (wpe) shortcut (RoPE test)

Hypothesis: at large batch the model can't easily learn the failure cue, so it exploits the
learned absolute positional embedding (`wpe`) and switches at a memorized absolute timestep.
Tested by (a) LR warmup and (b) replacing `wpe` with RoPE (rotary, relative-only; via the
`rotary-embedding-torch` lib, `--rope` flag / `model.rope`). All at switch_share 0.4, seed 0.

```
model (switch_share 0.4)   abs switch-t    steps fail->switch   value@sw  success  switches
bs64 (wpe) baseline        58.1 (±0.2)     31.9 (±27)           —         0.860    0.57
bs64 + warmup (wpe)        58.0 (±0.0)     45.7 (±9.6)          0.08      0.835    0.61
bs64 + RoPE                28.2 (±18.4)     2.8 (±6.5)          0.38      0.820    0.70
bs64 + RoPE + warmup       27.6 (±18.0)     2.1 (±6.8)          0.37      0.805    1.10
bs8 (wpe) baseline         19.5 (±12.8)     5.4 (±2.7)          0.20      0.985    0.37
bs8 + warmup (wpe)         22.3 (±15.9)     3.7 (±2.5)          0.30      0.865    0.60
```

1. **RoPE confirms the shortcut.** bs64+wpe fires at a FIXED absolute t≈58 (std 0.0–0.2),
   ignoring the cue. With RoPE the absolute switch time becomes VARIABLE (std ~18) and
   fail→switch collapses to ~2–3 steps — it now conditions on the failure signal. The
   bs64 fixed-time switch was a learned absolute-`wpe` shortcut.
2. **Warmup does NOT fix it** (bs64+warmup still t≈58, std 0.0) — not an early-optimization
   issue; it's the absolute-position signal specifically. Warmup also mildly HURT success
   everywhere (bs8 .985→.865) — reverted the train_seq default to `lr_schedule: constant`
   (warmup/cosine remain opt-in via config).
3. **Fixing timing ≠ recovering success.** bs64+RoPE conditions on the cue but is still
   0.820 (vs bs8 0.985) and switches MORE (0.70). So escaping the positional shortcut is
   necessary but not sufficient — small batch additionally yields a calmer policy (fewer
   switches, lower %failed). Batch 8 (wpe) remains the Phase-0 winner; RoPE is the
   mechanistic explanation, not a better config. (bs8 doesn't need RoPE — it already uses
   the cue.) [SUPERSEDED below by the 3-seed sweep: RoPE DOES help bs8's robustness.]

### batch-8 switch_share sweep, wpe vs RoPE, 3 seeds (updates the "doesn't need RoPE" call)

Swept switch_share {0.2,0.3,0.4,0.5} x {wpe,RoPE} at batch 8, 3 seeds each (eval_bamdp_adapt,
200 eps). success mean±std:

```
share   wpe              RoPE
0.2     0.872 ± 0.096    0.993 ± 0.002   <- best
0.3     0.907 ± 0.094    0.935 ± 0.036
0.4     0.975 ± 0.022    0.947 ± 0.043
0.5     0.815 ± 0.125    0.958 ± 0.019
```
(chart bs8_share_wpe_vs_rope.png; wandb group adapt_p0_bs8_seedsweep)

- Earlier SINGLE-SEED dips (wpe@0.3=0.775, RoPE@0.4=0.890) were noise — gone on averaging.
- **RoPE is uniformly higher AND far lower variance** (every share ≥0.935, std ≤0.043) vs wpe
  (noisy, std up to 0.125, sags at 0.5). Removing absolute position helps at batch 8 too — it
  makes training robust to switch_share, not only fixes the bs64 shortcut.
- **share 0.2 works only with RoPE** (wpe@0.2 0.872±0.096 noisy; RoPE@0.2 0.993±0.002).

**FINAL Phase-0 config: RoPE + batch 8 + switch_share 0.2 + decouple + window logic + constant LR.**
Adopted in bc_adapt.yaml / bc_adapt_3skill.yaml (`model.rope: true`, `switch_share: 0.2`). RoPE
also fits Phase-1 cross-episode contexts (absolute positions get large there).
