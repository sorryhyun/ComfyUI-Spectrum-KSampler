# Spectrum for ComfyUI

Training-free diffusion sampling acceleration via **Chebyshev polynomial feature forecasting** ([Han et al., CVPR 2026](https://arxiv.org/abs/2603.01623)). Drop-in KSampler replacement and standalone MODEL patcher that skip transformer blocks on predicted steps for ~2-3x speedup.

Tuned for the [Anima](https://github.com/sorryhyun/anima_lora) DiT — its modulation guidance, DCW calibrator artifacts, and per-block guidance presets are derived from the Anima training/inference pipeline. See the [anima_lora repo](https://github.com/sorryhyun/anima_lora) for the underlying methods documentation (`docs/methods/dcw.md`, `docs/methods/mod-guidance.md`, `docs/methods/smc_cfg.md`, `docs/methods/spectrum.md`).

## Contents

- [How it works](#how-it-works) — Chebyshev forecasting, adaptive window schedule, SEA scheduling
- [Usage](#usage) — node placement, sampler compatibility
  - [Standalone MODEL patcher](#standalone-model-patcher) — wire Spectrum before stock samplers
  - [DiT Spectrum Patch Advanced](#dit-spectrum-patch-advanced) — safer cache policy for complex sampler flows
  - [Standalone correction patchers](#standalone-correction-patchers) — DCW / SMC-CFG / CFG++ / FSG as MODEL patches
- [Parameters](#parameters) — Spectrum knobs + tuning tips
- [Modulation guidance](#modulation-guidance) — AdaLN-side quality steering via `pooled_text_proj`
- [SMC-CFG (α-adaptive sliding-mode CFG)](#smc-cfg-α-adaptive-sliding-mode-cfg) — velocity-space CFG combine modification
- [DCW post-step bias correction](#dcw-post-step-bias-correction) — sampler-level SNR-t bias correction
- [SPEED (Spectrum + SPD multi-resolution)](#speed-spectrum--spd-multi-resolution) — low-res prefix + spectral expansion, stacked on Spectrum
  - [Auto-scheduled LoRA-SPD node](#auto-scheduled-lora-spd-node) — loads an SPD-trained LoRA and reads its schedule from metadata

The feature blocks compose cleanly — they intervene at distinct points in the sampling loop (AdaLN → CFG combine → post-step x-space → and the SPEED node, which owns the loop).

## How it works

Standard diffusion runs the full DiT (all transformer blocks) at every denoising step. Spectrum observes that block outputs are smooth functions of the timestep, so most steps can be **predicted** instead of computed.

On "actual" steps the full model runs and block outputs are captured. On "cached" steps all transformer blocks are skipped — only `t_embedder` + `final_layer` + `unpatchify` execute, using features predicted from a Chebyshev ridge-regression fit.

### Adaptive window schedule

The window size N starts at `window_size` and grows by `flex_window` after each actual forward:

1. **Warmup** (first N steps): always run full forward to seed the forecaster
2. **Adaptive**: actual forward every `floor(N)` cached steps; N grows after each forward

With 28 steps and defaults: ~**8 actual forwards** out of 28 total steps.

### SEA scheduling (content-aware skip decision)

The growing window above is content-blind — it spends compute on a fixed cadence regardless of what the trajectory is doing. The **SEA** (Spectral-Evolution-Aware, SeaCache) schedule instead refreshes when the accumulated SEA-filtered latent distance crosses a calibrated threshold δ, so actual forwards land on the steps that actually move content.

The **KSampler (Spectrum)** node selects it via the `refresh_ratio` dial:

- `-1` → SEA off: the plain growing window above. Accelerates from the first run, no calibration.
- `0` (default) → SEA auto: δ is calibrated so the refresh fraction matches the growing window at this step count — **same speed**, smarter step placement.
- `>0` → explicit refresh ratio. Lower = fewer actual forwards = faster, less faithful.

The first run at each `(resolution / steps / cfg / refresh_ratio)` does a one-time full-compute calibration pass to fit δ, then caches it to the ComfyUI user dir; later runs at that config get the fast SEA trigger. δ generalizes across mod-guidance, so calibrating with mod active stays valid. SEA is incompatible with the SPEED/SPD node (mid-loop σ re-spacing breaks the distance trace) and transparently falls back to the window schedule there.

## Usage

Place the **KSampler (Spectrum)** node where you'd normally use a KSampler. It has the same inputs (model, seed, steps, cfg, sampler, scheduler, conditioning, latent) plus Spectrum-specific parameters. This single node now folds in what used to be three separate nodes:

- **Modulation guidance** — wire a `CLIP` and pick a `mod_w_profile` (default `step_i8_skip27`; `off` to disable). With no CLIP connected and a profile selected, mod guidance is skipped with a console warning rather than erroring — so a graph that only wires the sampler still runs.
- **SEA scheduling** — the `refresh_ratio` dial chooses *which* steps to skip: `0` (default) = SEA auto (content-aware, compute-matched to the growing window), `-1` = SEA off (plain growing window, accelerates from the first run with no calibration), `>0` = explicit refresh ratio.
- **α-adaptive SMC-CFG** — `adaptive_smc_alpha`.

> **Migration:** the former `KSampler (Spectrum + Mod Guidance)` and `KSampler (Spectrum SEA + Mod Guidance)` nodes are gone — their behavior is the default of this unified node. Their class keys remain as hidden aliases so existing saved workflows still load (they resolve to this node), but they no longer appear in the add-node menu. DCW + raw forecasting/guidance scalars still live on the **Advanced** node.

Works with any ComfyUI sampler (Euler, DPM, er_sde, etc.) because caching is handled transparently inside a model function wrapper. Chains with other model wrappers (Flex Attention, Flash Attention 4, etc.).

### Standalone MODEL patcher

The **DiT Spectrum Patch** node exposes Spectrum as a `MODEL → MODEL` patcher instead of a sampler. Wire it before ComfyUI's normal sampler nodes when you want to keep the stock sampling workflow:

```
CheckpointLoader / model loader → DiT Spectrum Patch → KSampler / KSampler Advanced / Custom Sampler
```

Set `steps` on **DiT Spectrum Patch** to the same value as the downstream sampler. The patcher keeps the original Spectrum path: actual steps capture the DiT feature immediately before `final_layer`; cached steps predict that feature and run only `final_layer` + `unpatchify`. It does **not** add modulation guidance, DCW, SMC-CFG, SPEED/SPD, noise generation, latent padding, or a custom sampling loop.

Use `enabled = false` to pass the input model through unchanged. If the same patched MODEL is connected to multiple sampler nodes, Spectrum applies to every sampler that consumes that MODEL output. Set `one_sampler_only = true` when the patch should apply only to the first sampler run; later sampler runs *within the same workflow run* (e.g. a hi-res-fix second pass) using the same patched MODEL pass through without Spectrum. This re-arms automatically on each new workflow execution, so re-queuing the graph applies Spectrum again. Non-DiT models fail with a clear error rather than silently producing invalid output.

### DiT Spectrum Patch Advanced

The **DiT Spectrum Patch Advanced** node is the same `MODEL -> MODEL` patcher with one extra control: `compat_policy`. Use it when the downstream sampler path is more complex than a single positive / single negative KSampler, for example:

- multi-positive conditioning
- exact artist-mix conditioning
- regional or masked conditioning
- Custom Sampler graphs
- chained model wrappers that may need to run on every model call

The node does not change sockets, conditioning data, sampling steps, or CFG math. It only decides whether a Spectrum step is allowed to use the fast cached prediction path. If the selected policy says the cache is unsafe, that step falls back to an actual DiT forward. Quality is protected by giving up speed for that branch or step.

For exact artist mixes, prefer:

```
MODEL -> DiT Spectrum Patch Advanced -> KSampler / KSampler Advanced / Custom Sampler
Artist Mixer exact positive -> sampler positive
negative -> sampler negative
```

Set the patcher's `steps` to the downstream sampler's `steps`, then start with `compat_policy = strict`. If you only need the old fastest behavior, use `legacy`. For a balanced default on wrapper-heavy but UUID-capable graphs, use `conservative`.

#### compat_policy modes

| Mode | Use when | Cached prediction is allowed when |
|------|----------|-----------------------------------|
| `legacy` | You want the original fastest Spectrum behavior and backwards-compatible saved workflows. | The normal Spectrum schedule says the step can be cached, unless an explicit cache veto callback blocks it. Existing wrappers may still be bypassed on cached steps, matching the old behavior. |
| `conservative` | You want safer behavior with custom samplers, shape changes, wrapper chains, or cache veto callbacks. | The normal Spectrum schedule allows caching and the runtime checks pass: valid batch split, matching latent shape, expected step count, no unsafe wrapper, and no veto callback blocking cache. Otherwise the step runs actual DiT forward. |
| `strict` | You are using exact artist mixes, multi-positive conditioning, or other flows where each conditioning branch must keep its own forecaster history. | Everything required by `conservative`, plus ComfyUI per-conditioning UUID branch keys must be available. If UUIDs are missing, it runs actual DiT forward instead of sharing a coarse cond/uncond cache. |

Why this matters: Spectrum forecasts DiT features from previous actual steps. In a simple prompt there is usually one positive branch and one negative branch. In exact artist mix, each artist is a separate positive conditioning branch. `strict` keeps those branch histories separate by requiring ComfyUI UUID keys before caching; without them, it avoids cached predictions rather than mixing unrelated artist trajectories.

### Standalone correction patchers

The extra correction features are also available as standalone `MODEL -> MODEL` patches. These patchers are tuned for Anima-style flow-matching DiTs. They may run on other DiT models that expose compatible ComfyUI sampler semantics, but quality and stability are not guaranteed there.

Recommended model chain:

```
MODEL
-> Anima Mod Guidance / LoRA patches
-> DiT CFG-FSG/DCW Patch
-> DiT Spectrum Patch Advanced
-> KSampler / KSampler Advanced / Custom Sampler
```

**DiT CFG-FSG/DCW Patch** groups DCW and the CFG/sigma-schedule corrections in one MODEL patcher:

- `dcw_mode` enables DCW post-step latent correction. `auto` requires the optional `clip` and `positive` inputs; if either is missing the node raises a clear error instead of silently falling back.
- `smc_cfg` enables SMC-CFG and requires the optional `cfg` input.
- `cfgpp` + `cfgpp_lambda` enables CFG++ and requires optional `steps`, `cfg`, `sampler_name`, `scheduler`, and `denoise`.
- `fsg` enables FSG fixed-point latent calibration and requires the same optional sampler inputs as CFG++.

SMC-CFG and CFG++ cannot both be enabled because both replace `sampler_cfg_function`. FSG is not allowed with SMC-CFG in this patcher; use FSG with CFG++ or, experimentally, plain CFG. When FSG is enabled, the patcher registers a Spectrum cache veto for FSG in-band steps. For that veto to work, place **DiT CFG-FSG/DCW Patch before DiT Spectrum Patch Advanced**.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 2.0 | Initial caching window N |
| `flex_window` | 0.25 | Window growth rate per actual forward |
| `warmup_steps` | 6 | Steps that always run full forward |
| `blend_w` | 0.3 | Chebyshev/Taylor blend weight (1.0 = pure Chebyshev) |
| `cheby_degree` | 3 | Number of Chebyshev basis functions |
| `ridge_lambda` | 0.1 | Ridge regression regularization strength |

Additional **DiT Spectrum Patch** parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps` | 30 | Must match the downstream sampler's step count |
| `tail_actual_steps` | 3 | Final steps that always run actual DiT forwards |
| `history_size` | 100 | Forecaster buffer size (same default as the integrated samplers) |
| `enabled` | true | `false` returns the input MODEL unchanged |
| `one_sampler_only` | false | Apply Spectrum only to the first sampler run that uses this patched MODEL |
| `verbose` | false | Logs actual/cached step decisions |

Additional **DiT Spectrum Patch Advanced** parameter:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compat_policy` | `legacy` | Cache-safety mode. `legacy` keeps the original fastest behavior. `conservative` uses cached predictions only when runtime checks say the step is safe. `strict` additionally requires ComfyUI per-conditioning UUIDs, which is the recommended mode for exact artist mixes and multi-positive conditioning. |

### Tuning tips

- **More speedup**: increase `flex_window` (faster window growth = fewer forwards)
- **Better quality**: increase `warmup_steps`, decrease `flex_window`
- **Aggressive acceleration**: `flex_window=1.0`, `blend_w=0.7` (~3-4x speedup)

## Modulation guidance

The **KSampler (Spectrum)** node (via `mod_w_profile`) and the **Advanced** node add text-conditioned quality steering via a learned `pooled_text_proj` MLP adapter ([Starodubcev et al., ICLR 2026](https://arxiv.org/abs/2502.15349)). The adapter projects pooled text embeddings into a guidance delta that is injected into the DiT's AdaLN timestep embedding, steering generation toward the specified quality attributes. The standalone **Anima Mod Guidance** patcher exposes the same profile surface as a `MODEL → MODEL` node so it composes with any sampler.

The default ~12MB `pooled_text_proj` weight is auto-downloaded on first use from the [anima_lora release page](https://github.com/sorryhyun/anima_lora/releases/tag/mod_guidance) into `ComfyUI/models/anima_mod_guidance/`. The unified node always uses the default; the advanced node exposes an adapter dropdown where `(auto-download default)` triggers the same download or you can pick a custom adapter from `loras/`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip` | — | CLIP encoder for encoding quality tags |
| `adapter` | `(auto-download default)` | `pooled_text_proj` safetensors file (advanced node only) |
| `quality_tags` | `absurdres, highres, masterpiece, ...` | Quality/aesthetic tags to steer toward (the steering axis's positive pole) |
| `quality_neg` | _(empty)_ | Negative pole of the steering axis (`delta = proj(quality_tags) − proj(quality_neg)`). **Empty reuses the CFG negative** (legacy behavior). Set a clean counter-pole like `worst quality, score_1` to decouple the quality axis from the broad CFG negative — the shared negative is anti-correlated (~−0.38 cosine) with the intended quality direction and ~40% weaker. Does **not** touch the CFG negative. |
| `mod_w_profile` (simple) | `step_i8_skip27` | Per-block guidance preset. `off` disables modulation guidance entirely (no adapter download, no extra hook). `step_i8_skip27` (default, best quality) protects blocks 0–7 + 27 and applies `w=3` to blocks 8–26. `step_i14` is the safe option — use it when a LoRA shows anatomy drift. `uniform_w3` recovers pre-0413 legacy behavior. |
| `mod_w` (advanced) | 3.0 | Peak guidance strength applied per-block |
| `mod_start_layer` (advanced) | 8 | First block (inclusive) that receives the steering delta. `0` = uniform legacy behavior |
| `mod_end_layer` (advanced) | -1 | Last block + 1 (exclusive). `-1` = all remaining blocks. Set to `27` to skip Anima's compensation block |
| `mod_taper` (advanced) | 0 | Number of late slots to scale by `mod_taper_scale`. `0` disables taper |
| `mod_taper_scale` (advanced) | 0.25 | Multiplier for tapered slots |
| `mod_final_w` (advanced) | 0.0 | `w` applied at `final_layer`. `0` = don't disturb the output head |

Per-block guidance schedules address quality drift on LoRAs whose distribution sits far from the positive-prompt axis (e.g. early blocks blowing out tonal DC into uniform color collapse). The default `step_i8_skip27` protects blocks 0–7 and the final compensation block 27 from the steering delta while keeping the base text projection uniform across all blocks. See [`docs/methods/mod-guidance.md`](https://github.com/sorryhyun/anima_lora/blob/main/docs/methods/mod-guidance.md) in anima_lora for the underlying rationale.

**Decoupled negative (`quality_neg`).** The mod-guidance delta is `proj(quality_tags) − proj(negative)`. By default the node bound that negative to the **CFG** negative — the broad kitchen-sink list (`worst quality, …, sepia, speech bubble, borders, …`) whose job is velocity-space repulsion, not to be a clean quality counter-pole. Geometrically the resulting axis is *anti-correlated* (cos ≈ −0.38) with the intended `quality_up − quality_down` direction and ~40% smaller in magnitude, so the push both misdirects and weakens. Set `quality_neg` to a focused counter-pole (e.g. `worst quality, score_1`) to recover a pure quality axis; the CFG negative keeps its broad role unchanged. The mod-guidance head is max-pooled and trained on max-pooled inputs — `quality_neg` does **not** change pooling. Leaving it empty reproduces the pre-decoupling result bit-for-bit. (The anima_lora CLI was already decoupled via `--mod_neg_prompt`; this brings the Comfy node in line.)

## SMC-CFG (α-adaptive sliding-mode CFG)

Every sampler node exposes an `adaptive_smc_alpha` knob (default `0.2`, set `0` to disable) that swaps the standard CFG combine for an **α-adaptive sliding-mode controller** ([Wang et al., arXiv:2603.03281](https://arxiv.org/abs/2603.03281), Anima α-adaptive form). The Advanced node additionally exposes `smc_cfg_lambda`, and **DiT CFG-FSG/DCW Patch** exposes the same correction as a standalone MODEL patch. Auto-skipped when CFG = 1 (no cond/uncond residual to slide on).

At each denoising step, with `e_t = v_cond − v_uncond`:

```
s_t        = (e_t − e_prev) + λ · e_prev      # sliding surface
k_t        = α · mean(|e_t|)                  # adaptive switching gain
Δe         = −k_t · sign(s_t)                 # bang-bang correction
v̂_t        = v_uncond + w · (e_t + Δe)        # modified CFG combine
```

The α-adaptive gain replaces the paper's fixed `k = 0.1`, which was ~14× too large for Anima at CFG=4 (see `bench/smc_cfg/analysis_and_proposal.md` in anima_lora — at the paper's `k` the bang-bang exceeded `|e|` at 93% of steps, turning the controller into a noise source). `k_t = α · mean(|e_t|)` self-scales across model / CFG / σ / sample.

Observable on Anima at CFG=4:

- **Detail / fine-structure recovery** — fingers, eyes, small text get sharper. High-`|e|` directions (clear semantic moves) are barely perturbed in relative terms; low-`|e|` per-voxel CFG noise gets clamped or sign-flipped.
- **Slight luminance drop** — outputs run a touch darker than vanilla CFG. The flow-matching DiT's `e` carries a small consistently-signed per-channel DC component (brightness/saturation lift of the conditional distribution) which is exactly the regime SMC clamps most aggressively. Lower `λ` attenuates the darkening while preserving most of the detail win.

| Parameter | Default | Description |
|---|---|---|
| `adaptive_smc_alpha` (all nodes) | 0.2 | α-adaptive switching gain. `0` disables (vanilla CFG combine). `0.2` puts the bang-bang correction at ~20% of the per-step mean residual magnitude. |
| `smc_cfg_lambda` (advanced) | 5.0 | Sliding-manifold slope λ. Paper sweep `{3,4,5,6}`; 5 best. Higher λ tightens the `sign()` pattern's grip on small-`|e|` channels (more detail recovery, more darkening). |

SMC-CFG operates in **velocity space** — it must, because σ varies per step and a denoised-space port would not be numerically equivalent. The ComfyUI hook converts via `v = (x_in − denoised) / σ`, runs the v-space combine, then converts back. One velocity-shaped buffer of state. No extra DiT forwards. Composes cleanly with mod guidance (AdaLN-side), DCW (post-step x-space), and Spectrum (cached steps still invoke the CFG combine). See [`docs/methods/smc_cfg.md`](https://github.com/sorryhyun/anima_lora/blob/main/docs/methods/smc_cfg.md) in anima_lora for the full derivation, the eps/tanh ablation, and composition table.

## DCW post-step bias correction

DCW is exposed on the **Advanced** sampler node and on **DiT CFG-FSG/DCW Patch**. It defaults to **off** — turn it on per-workflow when you want the correction. The unified **KSampler (Spectrum)** node does not run DCW; switch to **KSampler (Spectrum + Mod Guidance Advanced)** or chain **DiT CFG-FSG/DCW Patch** before your sampler to access the full `dcw_mode` / `dcw_lambda` / `dcw_band_mask` / `dcw_calibrator` surface. `dcw_mode = auto` runs the OnlineDCWCalibrator fusion head (per-prompt λ̂, auto-downloaded on first use) and requires the patch node's optional `clip` + `positive` inputs; `manual` uses the scalar `dcw_lambda × schedule(σ)`; `off` disables.

DCW ([Yu et al., CVPR 2026](https://arxiv.org/abs/2604.16044)) is a sampler-level post-step correction for the SNR-t bias of flow-matching DiTs. Each step's `prev_sample` is mixed toward (or away from) the post-CFG `x0_pred`, optionally restricted to a single-level Haar subband of the differential:

```
diff           = x_{i+1} − x0_pred_i
diff_masked    = haar_idwt(mask(haar_dwt(diff)))   # band restriction
x_{i+1}       += λ · (1 − σ_i) · diff_masked
```

| `dcw_lambda` | Behavior |
|---|---|
| `+0.01` (default) | Tuned for `dcw_band_mask = LL` at **CFG ≥ ~2** with non-square aspects. Positive (paper-direction) — recovers detail at non-square aspects without smoothing. |
| `-0.015` | Use at **CFG = 1 / 1024²**, where the bias direction flips and the paper-opposite sign applies. |
| `0.0` | Disabled — no overhead, no extra hooks registered. |
| Auto mode | Predicts a per-prompt λ̂ from the fusion head; ignores this widget. Tuned at CFG=4 — at CFG ≈ 1 prefer manual + `-0.015`. |

The bias direction is **(CFG × aspect)-dependent**. Default `+0.01` is the verified hyperparam for the common case (CFG=4, non-square). At CFG=1 / 1024² the optimal sign flips — drop to `-0.015` manually, or stick with auto if CFG ≥ ~2.

| `dcw_band_mask` | Behavior |
|---|---|
| `LL` (default) | Restrict correction to the Haar low-low subband. Strictly better than broadband on Anima — improves all four bands while broadband worsens the detail bands (LH/HL/HH). LL is the upstream causal lever; detail bands are downstream symptoms. |
| `all` | Paper-form broadband correction. Falls through to the cheap fused `add_` (no DWT round-trip). Pair with a smaller \|λ\| if you switch to this. |
| `HH`, `LH+HL+HH` | Ablation modes. `HH`-only is empirically dead; `LH+HL+HH` pulled in for completeness. |

The schedule is fixed to `one_minus_sigma` (correction concentrates at low σ where Anima's bias is largest). Implementation is sampler-agnostic — DCW mutates the latent at the step boundary via a `CALC_COND_BATCH` wrapper plus a post-CFG capture hook, so it composes correctly with Euler / ER-SDE / DPM++ / etc., with CFG on or off, and stacks cleanly on top of Spectrum + mod guidance. The DWT/iDWT round-trip on `LL` mode is one pass over the latent (negligible vs the DiT forward).

## SPEED (Spectrum + SPD multi-resolution)

The **KSampler (Spectrum + SPD / SPEED)** node stacks **SPD** (Spectral Progressive Diffusion — [Xiao et al., arXiv:2605.18736](https://arxiv.org/abs/2605.18736)) on top of Spectrum. SPD runs the early, noise-dominated steps at a **lower resolution** (cheap, far fewer tokens), then spectral-expands the latent to full resolution once finer frequencies emerge — and Spectrum forecasts the full-res tail. The two accelerations target different parts of the trajectory (SPD shrinks tokens early, Spectrum skips blocks late), so they compose rather than compete.

This is the **naive-reset compose** validated in [`bench/spd/compose_report.md`](https://github.com/sorryhyun/anima_lora/blob/main/bench/spd/compose_report.md): the low-res prefix runs uncached, then at the handoff Spectrum's forecaster is reset and re-warms over the full-res tail (phase-2-only). At the validated knee it was coherent across seeds and the fastest of the tested configs (≈×1.98 wall vs Spectrum-alone ≈×1.73).

| Parameter | Default | Description |
|---|---|---|
| `split_mode` | `single` | Resolution schedule. `single` = one low→full transition (v0). |
| `spd_scale` | `0.5` | Prefix resolution (fraction of full latent H/W). `0.5` ≈ a quarter the tokens — the benched-coherent point. `1.0` disables SPD (= vanilla Spectrum). |
| `spd_sigma` | `0.7` | Handoff σ: switch low→full when the schedule drops to this noise level. `0.7` is the validated knee. `1.0` disables SPD. |

**Caveats.** SPD re-spaces the σ schedule mid-loop, so the SPEED sampler is **Euler-only** (any other `sampler_name` is ignored with a warning). The scale `0.5` / σ `0.7` point is the only benched-coherent config — **lower scales or later/more-aggressive knees are untested** and stress the handoff re-warm (especially on HF-detail prompts). The reported block-skip ratio in the log understates the true speedup because it counts the cheaper low-res prefix forwards as full forwards.

### Auto-scheduled LoRA-SPD node

The **KSampler (SPD LoRA / auto-schedule)** node is for adapters distilled by the SPD trajectory-adapter workflow (`anima_lora` `make exp-spd`). Such a LoRA is fit to a specific resolution schedule and snapshots it into its safetensors metadata (`ss_spd_stages` / `ss_spd_transition_sigmas`). This node **loads the LoRA onto the MODEL itself** (pick it from the `lora_name` dropdown, `lora_strength` weights it) and reads that metadata to drive the SPEED sampler automatically — so inference geometry matches what the adapter trained on, with no manual scale/σ tuning. There are no schedule knobs by design.

| Parameter | Default | Description |
|---|---|---|
| `lora_name` | — | The SPD-trained LoRA. Its schedule is read from the file metadata. |
| `lora_strength` | `1.0` | LoRA weight multiplier applied to the MODEL. |

Unlike the base SPEED node, this node honors **multi-stage** schedules (e.g. `[0.5, 0.75, 1.0]` with `[0.7, 0.4]`): it spectral-expands at each handoff and only arms Spectrum once the trajectory reaches full resolution. If the selected LoRA happens to carry no schedule metadata, it falls back to the validated `0.5` / σ`0.7` single handoff (with a warning). To stack a style LoRA or mod guidance, chain a stock `LoraLoader` / **Anima Mod Guidance** node upstream on the MODEL input. Euler-only and the SPEED caveats above still apply.

### Modulation guidance as a standalone patcher

The **Anima Mod Guidance (model patch)** node applies the same quality steering as the mod-guidance samplers, but as a standalone `MODEL → MODEL` patcher. Wire its output into *any* sampler — including the SPEED node — to compose mod guidance with SPD without a dedicated combined node. (Wire the same `positive` / `negative` conditioning into both the patcher and the sampler; the patcher reads them only to compute the steering delta.) The in-sampler mod-guidance nodes are unchanged.

## CNS — Colored Noise Sampling (sampler entry, not a node)

CNS ([Davidson et al., arXiv:2605.30332](https://arxiv.org/abs/2605.30332)) is a training-free SDE plug-in: it replaces the **white** noise the ER-SDE solver injects each step with **frequency-colored** noise that dumps the step's fixed stochastic-energy budget into the radial frequency bands the network has *not yet resolved* at that σ (per a precomputed completion matrix γ(f, t)). It is a zero-sum spectral *reallocation* of a fixed variance budget — RMS-renormalized, not a noise scale-up — so it sharpens late high-frequency detail without pushing off-manifold.

Unlike the other features here, CNS ships as a **sampler-dropdown entry**, not a node: it lives at the per-step noise-injection seam, which no model patch can reach. Just pick **`er_sde_cns`** in any KSampler's `sampler_name` field.

```
sampler_name = er_sde_cns        # ER-SDE solver + CNS-recolored per-step noise
```

- **er_sde only.** CNS only acts on the stochastic path — it *is* the ER-SDE solver with recolored injection. There is no euler/ODE surface (white draw → nothing to recolor). With `er_sde_cns` you get ER-SDE + recoloring in one pick.
- **Anima-calibrated, auto-downloaded.** The shipped γ matrix (cfg=4, Anima spectral-bias staircase; ~6 KB) auto-downloads to `models/anima_cns/` on first use. On a non-Anima model it is mis-calibrated (still variance-conserving, so degraded-not-broken) — the global dropdown gives no per-model signal, so use it on Anima workflows.
- **Composes.** Recoloring is the noise term; it stacks cleanly with the Spectrum wrapper and the DCW / SMC-CFG / mod-guidance model patches (all different seams). Full strength, no knobs (a strength < 1 white-blend is strictly inferior per the paper's ablation, so it is not exposed).
