# MOSS-Image: ComfyUI schedulers vs the pipeline's σ grid

Applies to **KSampler (Spectrum for MOSS)** and to any ComfyUI workflow sampling MOSS-Image.
Measured 2026-10-25 in anima_lora (`project/moss_lora/src/schedule_ab.py`).

## The grids

The reference pipeline (`pipeline_moss_image.py`, `MOSSImageScheduler`) builds 50 points
from t = 1 down to t = 5/1004, applies shift 5, and snaps each σ to the nearest point of the
1000-step training grid. ComfyUI's schedulers do not snap.

| scheduler | relation to the pipeline grid | last σ before 0, 25 steps | 50 steps |
|---|---|---|---|
| pipeline (snapped) | — | 0.024 | 0.024 |
| `normal` | the same grid without the snap; within 0.0005 σ at every step | 0.024 | 0.024 |
| `sgm_uniform` | within 0.0065 σ above σ 0.8, stops earlier | 0.190 | 0.113 |
| `simple` | within 0.0049 σ above σ 0.8, stops earlier | 0.172 | 0.093 |

`normal` matches because ComfyUI's `sigma_min` for shift 5 is shift(1/1000) = 5/1004, the
pipeline's lower bound.

Composition forms above σ ≈ 0.8, which is the first 28 of 50 steps. In that range all
four grids are within 0.0065 σ of each other. They differ mostly in the tail.

Which workflow uses which scheduler:

- this node and the OpenMOSS plugin workflow: `sgm_uniform`, 50 steps
- Comfy-Org's native templates: `simple`, 25 steps

## A/B

Setup: MOSS-Image Base, Euler, CFG 3, uncond skip 22, empty negative, 1024², seed 1, no
LoRA. Three prompts (`anime_sunset`, `zh_fridge`, `ja_manga`), 50 steps. Each schedule ran
twice: every step actual (p50), and with Spectrum at this node's defaults (s50).

**No schedule gave visibly better images.** All four rendered the fridge note's Chinese and
the manga bubble's 「待ってくれ！」 correctly. Hair, eye and line detail looked the same in
native-resolution crops. A last σ of 0.024 against 0.11 left no visible residual noise or
softness. The one defect was a stray stroke after `面包` on `sgm_uniform`, at one seed.

**The schedules differ the way different samples do.** Here is the latent rel-L2 of each
p50 run against the snapped p50:

| | anime_sunset | zh_fridge | ja_manga |
|---|---|---|---|
| `normal` | 0.106 | 0.241 | 0.307 |
| `sgm_uniform` | 0.364 | 0.327 | 0.390 |
| `simple` | 0.074 | 0.413 | 0.383 |

`normal` is 10× closer to the snapped grid in σ than `simple`. Even so, `simple` came
closest on `anime_sunset`: hand position, finger placement and the fold of the fridge note
match the snapped image best there. A change of under 0.0065 σ during composition is enough
to pick a different sample. Which way it goes does not follow σ distance.

## Spectrum on these schedules

Spectrum's full-step schedule (warmup 6, window 2, `flex_window` 0.5, tail 3) was tuned on
the snapped grid. Its window and forecaster count steps, not σ, so the full forwards land on
the same step indices under any scheduler. Above σ 0.8 those steps sit at nearly the same σ.

Drift of s50 from the same schedule's p50, latent rel-L2:

| | anime_sunset | zh_fridge | ja_manga | mean |
|---|---|---|---|---|
| snapped | 0.359 | 0.481 | 0.542 | 0.460 |
| `normal` | 0.358 | 0.498 | 0.532 | 0.463 |
| `sgm_uniform` | 0.349 | 0.477 | 0.494 | 0.440 |
| `simple` | 0.315 | 0.509 | 0.511 | 0.445 |

The drift is the same on all four, so the defaults carry over to `sgm_uniform`. Spectrum
moves the image more than the choice of schedule does: on `zh_fridge` every s50 run drew a
smaller note than its p50 run.

## Not tested

- Fewer steps. At 25 steps the tails differ more (0.024 against 0.17–0.19). Comfy-Org's
  templates run there.
- More seeds, judged blind. One seed can show a sample difference, not a quality ranking.
- SFT.
