"""ComfyUI node definitions for Spectrum inference acceleration."""

import copy
import json
import logging
import math

import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths

from .dcw import install_dcw
from .dcw_calibrator import setup_dcw_calibrator
from .fsg import FSGCalibrator, fsg_step_indices, install_cfgpp, install_fsg
from .mod_guidance import AUTO_ADAPTER_SENTINEL, setup_mod_guidance
from .smc_cfg import install_smc_cfg
from .spectrum import COMPAT_POLICIES, apply_dit_spectrum_patch, spectrum_sample

logger = logging.getLogger(__name__)

AUTO_CALIBRATOR_SENTINEL = "(auto-download default)"


def _adapter_choices():
    return [AUTO_ADAPTER_SENTINEL] + folder_paths.get_filename_list("loras")


def _calibrator_choices():
    # Re-uses the loras directory for user-supplied artifacts (same convention
    # as mod-guidance adapters). Auto-download lands in models/anima_dcw_calibrator/.
    return [AUTO_CALIBRATOR_SENTINEL] + folder_paths.get_filename_list("loras")


# ---------------------------------------------------------------------------
# Common input definitions
# ---------------------------------------------------------------------------

_KSAMPLER_INPUTS = {
    "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
    "seed": (
        "INT",
        {
            "default": 0,
            "min": 0,
            "max": 0xFFFFFFFFFFFFFFFF,
            "control_after_generate": True,
        },
    ),
    "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
    "cfg": (
        "FLOAT",
        {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
    ),
    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
    "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
    "positive": ("CONDITIONING",),
    "negative": ("CONDITIONING",),
    "latent_image": ("LATENT",),
    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
}

_QUALITY_TAGS_INPUT = (
    "STRING",
    {
        "default": "highres, best quality, score_7",
        "multiline": True,
        "dynamicPrompts": True,
        "tooltip": "Quality tags to steer generation toward via modulation.",
    },
)

# Per-block guidance profiles. Each maps to a fixed (w, start, end, taper, taper_scale, final_w)
# tuple — see docs/mod-guidance.md for the naming + rationale.
# end_layer = -1 means "all blocks" (resolved against num_blocks at setup time).
MOD_W_PROFILE_OFF = "off"
MOD_W_PROFILES = {
    "step_i8_skip27": dict(
        w=3.0, start_layer=8, end_layer=27, taper=0, taper_scale=0.25, final_w=0.0
    ),
    "step_i14": dict(
        w=3.0, start_layer=14, end_layer=-1, taper=0, taper_scale=0.25, final_w=0.0
    ),
    "uniform_w3": dict(
        w=3.0, start_layer=0, end_layer=-1, taper=0, taper_scale=0.25, final_w=0.0
    ),
}
DEFAULT_MOD_W_PROFILE = "step_i8_skip27"


_CLIP_INPUT = {
    "clip": ("CLIP", {"tooltip": "CLIP encoder for encoding positive quality tags."}),
}

_QUALITY_NEG_INPUT = (
    "STRING",
    {
        "default": "score_1, score_2, score_3, worst quality, lowres, old, bad hands, bad anatomy",
        "multiline": True,
        "dynamicPrompts": True,
        "tooltip": (
            "Quality-negative baseline for the mod-guidance steering axis "
            "(delta = proj(quality_tags) − proj(quality_neg)). Leave EMPTY to "
            "reuse the CFG negative (legacy behavior). Set a clean counter-pole "
            "(e.g. 'worst quality, score_1') to decouple the quality axis from "
            "the broad CFG negative, which is anti-correlated with the intended "
            "quality direction. Does NOT change the CFG negative itself."
        ),
    },
)

_MOD_PROFILE_INPUTS = {
    "quality_tags": _QUALITY_TAGS_INPUT,
    "quality_neg": _QUALITY_NEG_INPUT,
    "mod_w_profile": (
        [MOD_W_PROFILE_OFF] + list(MOD_W_PROFILES.keys()),
        {
            "default": DEFAULT_MOD_W_PROFILE,
            "tooltip": (
                "Per-block guidance schedule preset. "
                "'off' disables modulation guidance entirely (no adapter download, no extra hook). "
                "'step_i8_skip27' (default) protects early tonal-DC blocks 0–7 and the "
                "final compensation block 27, applying w=3 to blocks 8–26 — best overall "
                "quality but can occasionally show minor anatomy drift on drift-prone LoRAs. "
                "'step_i14' is the SAFE option: steers only from block 14 onward, reliably "
                "stays inside the trained manifold at the cost of a slightly less expressive result. "
                "'uniform_w3' recovers pre-0413 behavior (not recommended — prone to pink-collapse)."
            ),
        },
    ),
}


def _mod_guidance_advanced_inputs():
    return {
        "clip": _CLIP_INPUT["clip"],
        "adapter": (
            _adapter_choices(),
            {
                "tooltip": (
                    "pooled_text_proj safetensors adapter. "
                    f"'{AUTO_ADAPTER_SENTINEL}' fetches the default ~12MB weight "
                    "from the anima_lora release page on first use."
                ),
            },
        ),
        "quality_tags": _QUALITY_TAGS_INPUT,
        "quality_neg": _QUALITY_NEG_INPUT,
        "mod_w": (
            "FLOAT",
            {
                "default": 3.0,
                "min": -20.0,
                "max": 20.0,
                "step": 0.1,
                "tooltip": "Peak modulation guidance strength applied per-block.",
            },
        ),
        "mod_start_layer": (
            "INT",
            {
                "default": 8,
                "min": 0,
                "max": 999,
                "tooltip": (
                    "Inclusive first block index that receives the steering delta. "
                    "0 = uniform (pre-0413). 8 = protect tonal-DC blocks 0–7 (default)."
                ),
            },
        ),
        "mod_end_layer": (
            "INT",
            {
                "default": 27,
                "min": -1,
                "max": 999,
                "tooltip": (
                    "Exclusive last block + 1. 27 (default, Anima has 28 blocks) "
                    "skips the final compensation block — matches the 'step_i8_skip27' "
                    "preset + the anima_lora CLI. -1 = all remaining blocks."
                ),
            },
        ),
        "mod_taper": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 999,
                "tooltip": (
                    "Number of late slots inside [start, end) to scale by taper_scale. "
                    "0 disables taper. 2 + end=27 reproduces the 'piecewise' preset."
                ),
            },
        ),
        "mod_taper_scale": (
            "FLOAT",
            {
                "default": 0.25,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "tooltip": "Multiplier applied to tapered slots (e.g. 0.25 -> w*0.25).",
            },
        ),
        "mod_final_w": (
            "FLOAT",
            {
                "default": 0.0,
                "min": -20.0,
                "max": 20.0,
                "step": 0.1,
                "tooltip": (
                    "w applied at final_layer. 0.0 = don't disturb the output head (default). "
                    "Set non-zero only if your LoRA needs final-layer steering."
                ),
            },
        ),
    }


_SPECTRUM_INPUTS = {
    "window_size": (
        "FLOAT",
        {
            "default": 2.0,
            "min": 1.0,
            "max": 10.0,
            "step": 0.25,
            "tooltip": "Initial caching window N — actual forward every floor(N) steps.",
        },
    ),
    "flex_window": (
        "FLOAT",
        {
            "default": 0.25,
            "min": 0.0,
            "max": 2.0,
            "step": 0.05,
            "tooltip": "Window growth rate — N increases by this after each actual forward.",
        },
    ),
    "warmup_steps": (
        "INT",
        {
            "default": 6,
            "min": 0,
            "max": 50,
            "tooltip": "Number of initial steps that always run actual forwards.",
        },
    ),
    "blend_w": (
        "FLOAT",
        {
            "default": 0.3,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "tooltip": "Chebyshev/Taylor blend weight (1.0 = pure Chebyshev).",
        },
    ),
    "cheby_degree": (
        "INT",
        {
            "default": 3,
            "min": 1,
            "max": 10,
            "tooltip": "Number of Chebyshev basis functions.",
        },
    ),
    "ridge_lambda": (
        "FLOAT",
        {
            "default": 0.1,
            "min": 0.001,
            "max": 10.0,
            "step": 0.01,
            "tooltip": "Ridge regression regularization strength.",
        },
    ),
}

_SPECTRUM_DEFAULTS = dict(
    window_size=2.0,
    flex_window=0.25,
    warmup_steps=6,
    blend_w=0.3,
    cheby_degree=3,
    ridge_lambda=0.1,
)

_COMPAT_POLICY_INPUT = (
    list(COMPAT_POLICIES),
    {
        "default": "legacy",
        "tooltip": (
            "How safely Spectrum is allowed to skip DiT blocks. 'legacy' is the "
            "fastest old behavior. 'conservative' runs an actual DiT forward "
            "instead of a cached prediction when wrappers, latent shape, step "
            "count, or veto checks look unsafe. 'strict' is the safest choice "
            "for exact artist/multi-conditioning mixes: it also requires "
            "ComfyUI per-conditioning UUIDs before caching."
        ),
    },
)

# SPD / SPEED: multi-resolution progressive diffusion composed with Spectrum
# (naive-reset compose, validated in anima_lora/bench/spd/compose_report.md).
# Low-res prefix runs uncached; at σ=spd_sigma the latent is spectral-expanded
# to full res and Spectrum forecasts the tail. Euler-only.
_SPD_INPUTS = {
    "split_mode": (
        ["single"],
        {
            "default": "single",
            "tooltip": (
                "SPD resolution schedule. 'single' = one low→full transition "
                "(v0). The low-res prefix runs at spd_scale, then expands to full "
                "resolution at σ=spd_sigma."
            ),
        },
    ),
    "spd_scale": (
        "FLOAT",
        {
            "default": 0.5,
            "min": 0.25,
            "max": 1.0,
            "step": 0.05,
            "round": 0.01,
            "tooltip": (
                "Prefix resolution scale (fraction of full latent H/W) for the "
                "low-res phase. 0.5 = quarter the tokens, the benched-coherent "
                "point. 1.0 disables SPD (vanilla Spectrum). Lower than 0.5 is "
                "untested and stresses the handoff re-warm."
            ),
        },
    ),
    "spd_sigma": (
        "FLOAT",
        {
            "default": 0.7,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "round": 0.001,
            "tooltip": (
                "Handoff σ: switch from low-res to full-res when the schedule "
                "drops to this noise level. 0.7 is the validated knee (re-warm "
                "overlaps Spectrum's natural warmup, so it's cheap). Later/lower "
                "knees and HF-detail prompts are untested. 1.0 disables SPD."
            ),
        },
    ),
}

# LoRA-SPD: an SPD-trained LoRA snapshots the resolution schedule it was fit to
# into safetensors metadata (ss_spd_stages / ss_spd_transition_sigmas, written by
# anima_lora/scripts/distill_spd.py). Reading it back lets the node auto-configure
# the SPEED sampler so train/infer geometry stays aligned with no manual tuning.
# If a selected LoRA carries no schedule metadata, fall back to the validated
# single-handoff knee rather than erroring.
SPD_FALLBACK_SCALE = 0.5
SPD_FALLBACK_SIGMA = 0.7


def _read_spd_schedule_meta(lora_name):
    """Read (stages, transition_sigmas, label) from an SPD LoRA's metadata.

    Returns ``(None, None, None)`` if the file has no schedule metadata or can't
    be read (e.g. a non-safetensors LoRA) — the caller then falls back to the
    manual scalars rather than hard-erroring.
    """
    from safetensors import safe_open

    lora_path = folder_paths.get_full_path("loras", lora_name)
    if lora_path is None:
        return None, None, None
    try:
        with safe_open(lora_path, framework="pt") as f:
            md = f.metadata() or {}
    except Exception as e:  # not safetensors / unreadable header
        logger.warning("SPD LoRA: could not read metadata from %s: %s", lora_name, e)
        return None, None, None

    label = md.get("ss_spd_schedule_label")
    raw_stages = md.get("ss_spd_stages")
    raw_trans = md.get("ss_spd_transition_sigmas")
    try:
        stages = [float(v) for v in json.loads(raw_stages)] if raw_stages else None
        trans = [float(v) for v in json.loads(raw_trans)] if raw_trans else None
    except (ValueError, TypeError) as e:
        logger.warning("SPD LoRA: malformed schedule metadata in %s: %s", lora_name, e)
        return None, None, label
    return stages, trans, label


# SMC-CFG: α-adaptive sliding-mode CFG combine (Wang et al., arXiv:2603.03281,
# Anima α-adaptive form — see anima_lora/docs/methods/smc_cfg.md). Replaces the
# vanilla `uncond + w·(cond-uncond)` combine; no extra forwards. α=0 disables.
_SMC_CFG_LAMBDA_DEFAULT = 5.0
_SMC_CFG_ALPHA_DEFAULT = 0.1

_SMC_CFG_ALPHA_INPUT = (
    "FLOAT",
    {
        "default": _SMC_CFG_ALPHA_DEFAULT,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "round": 0.001,
        "tooltip": (
            "α-adaptive Sliding-Mode Control CFG gain. "
            "0 disables (vanilla CFG combine). 0.2 = production default — "
            "k_t := α·mean(|v_cond − v_uncond|) per step keeps the bang-bang "
            "correction in-band across CFG/σ/sample (paper's fixed k=0.1 was "
            "~14× off on Anima at CFG=4). Recovers detail (fingers, eyes, text); "
            "outputs run slightly darker. Auto-disabled when CFG=1."
        ),
    },
)
_SMC_CFG_LAMBDA_INPUT = (
    "FLOAT",
    {
        "default": _SMC_CFG_LAMBDA_DEFAULT,
        "min": 0.0,
        "max": 20.0,
        "step": 0.5,
        "round": 0.1,
        "tooltip": (
            "SMC-CFG sliding-manifold slope λ. Paper sweep {3,4,5,6}; 5 best. "
            "Higher λ tightens the sign() pattern's grip on small-|e| channels "
            "(more detail recovery, more darkening); lower λ attenuates both."
        ),
    },
)

# DCW: SNR-t bias correction (arXiv:2604.16044). Anima form, λ < 0,
# schedule fixed to one_minus_sigma. See anima_lora/docs/methods/dcw.md.
# Exposed on the Advanced sampler and the standalone DiT CFG-FSG/DCW patcher.
_DCW_INPUTS = {
    "dcw_mode": (
        ["off", "manual", "auto"],
        {
            "default": "off",
            "tooltip": (
                "DCW correction mode. "
                "'off' disables correction entirely. "
                "'manual' uses the scalar dcw_lambda × schedule(σ) — predictable, "
                "user-tunable. "
                "'auto' uses an OnlineDCWCalibrator fusion head (~few MB, "
                "auto-downloaded on first use) to predict per-prompt λ̂ from "
                "warmup observations of the post-CFG velocity. Forces band='LL'. "
                "Tuned at CFG=4 — at CFG≈1 the head's α̂ direction may "
                "overshoot; prefer manual then."
            ),
        },
    ),
    "dcw_lambda": (
        "FLOAT",
        {
            "default": 0.01,
            "min": -1.0,
            "max": 1.0,
            "step": 0.001,
            "round": 0.0001,
            "tooltip": (
                "DCW post-step bias correction strength (manual mode). 0.0 = "
                "disabled. Default +0.01 is the verified hyperparam for "
                "LL-only at CFG≥~2 (recovers detail at non-square aspects). "
                "At CFG=1 / 1024² use ≈ -0.015. Schedule fixed to "
                "one_minus_sigma. Composes with Spectrum + mod guidance; "
                "sampler-agnostic. Ignored in auto mode."
            ),
        },
    ),
    "dcw_band_mask": (
        ["LL", "all", "HH", "LH+HL+HH"],
        {
            "default": "LL",
            "tooltip": (
                "Restrict DCW correction to a subset of single-level Haar "
                "subbands (manual mode). 'LL' (default) is strictly better "
                "than broadband on Anima — improves all four bands while "
                "'all' worsens detail bands. 'all' = paper-form broadband "
                "correction. 'HH' / 'LH+HL+HH' are ablation modes. "
                "Forced to 'LL' in auto mode."
            ),
        },
    ),
    "dcw_calibrator": (
        _calibrator_choices(),
        {
            "tooltip": (
                "Fusion-head safetensors artifact (auto mode only). "
                f"'{AUTO_CALIBRATOR_SENTINEL}' fetches the default head from "
                "the ComfyUI-Spectrum-KSampler release page on first use "
                "(stored under models/anima_dcw_calibrator/). Custom artifacts "
                "are picked up from the loras directory."
            ),
        },
    ),
}


# Foresight Guidance (FSG) + CFG++ substrate (paper arXiv 23177). FSG reframes
# CFG as a fixed-point calibration; faithful FSG is defined on a CFG++ substrate.
# The validated production point (1024 tier / 28-step er_sde) is the defaults
# below. Both need CFG != 1; neither composes with SPD/SPEED.
_FSG_INPUTS = {
    "cfgpp_lambda": (
        "FLOAT",
        {
            "default": 0.0,
            "min": 0.0,
            "max": 8.0,
            "step": 0.1,
            "round": 0.001,
            "tooltip": (
                "CFG++ substrate strength λ (0 = off, plain CFG). Replaces the "
                "constant-w cond/uncond combine with the σ-scheduled CFG++ weight "
                "(paper App A.2) — the substrate faithful FSG is defined on. "
                "λ=1.5 is the production point (tracks CFG=4 saturation/contrast/"
                "composition; <1.5 under-guides, >=2 over-saturates). This is a "
                "FLOW-space coefficient, not the paper's DDIM λ. Mutually exclusive "
                "with SMC-CFG (SMC wins if both on)."
            ),
        },
    ),
    "fsg": (
        "BOOLEAN",
        {
            "default": False,
            "tooltip": (
                "Foresight Guidance: pre-step latent calibration toward the golden "
                "path. Runs K forward-backward fixed-point iterations on the latent "
                "before each in-band step (each forced to an actual Spectrum "
                "forward; ~3·K extra forwards per in-band step). Needs CFG != 1. "
                "Pair with cfgpp_lambda=1.5 for the validated fsg/cfg++ point."
            ),
        },
    ),
    "fsg_band_lo": (
        "FLOAT",
        {
            "default": 0.59,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "tooltip": (
                "FSG σ-band lower bound. The band is where calibration fires. "
                "Default [0.59, 0.75] is the 1024-token-tier / 28-step er_sde "
                "point. The contracting band moves DOWN for more steps and for "
                "low-token (~768px) renders, UP for fewer steps. σ≈0.94 always "
                "DIVERGES (the paper's noisy-stage prescription is wrong on Anima) "
                "— do not raise hi past ~0.85. Re-tune if you change steps/res."
            ),
        },
    ),
    "fsg_band_hi": (
        "FLOAT",
        {
            "default": 0.75,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "tooltip": (
                "FSG σ-band upper bound. See fsg_band_lo. Default 0.75 is the "
                "28-step er_sde sweet spot (it was 0.85 at 20-step Euler — the "
                "band slid down on the denser grid)."
            ),
        },
    ),
    "fsg_k": (
        "INT",
        {
            "default": 3,
            "min": 0,
            "max": 8,
            "tooltip": (
                "FSG fixed-point iterations per in-band step (0 = inert). Error "
                "~ρ^K with ρ≈0.93, so K=3 captures ~all the gain; K=2 drift-"
                "saturates, K=3 adds visible detail. Each iteration is ~3 extra "
                "DiT forwards."
            ),
        },
    ),
    "fsg_d_sigma": (
        "FLOAT",
        {
            "default": 0.1,
            "min": 0.01,
            "max": 0.3,
            "step": 0.01,
            "round": 0.001,
            "tooltip": (
                "FSG forward-backward stride Δσ. Stability is governed by γ·Δσ; "
                "0.1 contracts at γ≈4. Larger Δσ is what makes the operator "
                "diverge — leave at 0.1 unless you also shrink γ."
            ),
        },
    ),
    "fsg_gamma": (
        "FLOAT",
        {
            "default": 0.0,
            "min": 0.0,
            "max": 16.0,
            "step": 0.5,
            "tooltip": (
                "FSG calibration guidance γ (0 = use the CFG scale). Keep ≈ the "
                "CFG scale (=4) even on the CFG++ substrate — matching γ to the "
                "CFG++ effective weight (~11 in-band) makes the operator DIVERGE "
                "(stability is set by γ·Δσ)."
            ),
        },
    ),
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _clone_model_options(model):
    """Copy model_options so model patch nodes do not mutate the input MODEL."""
    try:
        model.model_options = copy.deepcopy(model.model_options)
    except Exception as e:
        logger.warning(
            "DiT model patch: deepcopy(model_options) failed (%s); using a shallow copy.",
            e,
        )
        model.model_options = dict(model.model_options)


def _missing_option_error(feature_name, missing, *, hint):
    raise RuntimeError(
        f"{feature_name}: missing required optional input(s): "
        + ", ".join(missing)
        + f". {hint}"
    )


def _require_option_values(feature_name, *, hint, **values):
    missing = [name for name, value in values.items() if value is None]
    if missing:
        _missing_option_error(feature_name, missing, hint=hint)


def _sampler_cfg_function_exists(model):
    return model.model_options.get("sampler_cfg_function") is not None


def _require_sampler_cfg_slot(model, feature_name, replace_existing_cfg):
    if not _sampler_cfg_function_exists(model) or replace_existing_cfg:
        return
    raise RuntimeError(
        f"{feature_name} needs sampler_cfg_function, but this MODEL already has "
        "one. Disable the existing CFG patch or set replace_existing_cfg=true."
    )


def _add_spectrum_cache_veto(model, veto):
    """Register a cache-veto callback consumed by DiT Spectrum Patch Advanced."""
    existing = model.model_options.get("spectrum_cache_vetoes")
    if existing is None:
        vetoes = []
    elif callable(existing):
        vetoes = [existing]
    else:
        try:
            vetoes = list(existing)
        except TypeError:
            vetoes = []
    vetoes.append(veto)
    model.model_options["spectrum_cache_vetoes"] = vetoes


def _fsg_spectrum_cache_veto(fsg_steps):
    fsg_steps = frozenset(fsg_steps)

    def veto(state, **_kwargs):
        return getattr(state, "step_idx", -1) not in fsg_steps

    veto.__name__ = "dit_cfg_fsg_spectrum_cache_veto"
    return veto


def _apply_mod_guidance(
    model,
    clip,
    positive,
    negative,
    adapter,
    quality_tags,
    *,
    quality_neg="",
    w,
    start_layer,
    end_layer,
    taper,
    taper_scale,
    final_w,
):
    """Clone `model` and install the mod-guidance hook with explicit scalars."""
    m = model.clone()
    setup_mod_guidance(
        m,
        clip,
        positive,
        negative,
        adapter,
        quality_tags,
        w,
        quality_neg=quality_neg,
        start_layer=start_layer,
        end_layer=end_layer,
        taper=taper,
        taper_scale=taper_scale,
        final_w=final_w,
    )
    return m


def _apply_mod_profile(
    model, clip, positive, negative, quality_tags, profile_name, quality_neg=""
):
    """Patch `model` with the named per-block guidance profile.

    Returns the model unchanged when the profile is 'off'. If a profile is
    selected but no CLIP is connected (e.g. a pre-unification workflow that only
    wired the plain Spectrum sampler), mod guidance is skipped with a warning
    rather than hard-erroring — so old graphs keep behaving like the basic node.
    `quality_neg` (empty = reuse the CFG negative) decouples the steering axis.
    """
    if profile_name == MOD_W_PROFILE_OFF:
        return model
    if clip is None:
        logger.warning(
            "mod_w_profile=%r but no CLIP is connected — skipping mod guidance. "
            "Wire a CLIP encoder to enable quality steering, or set "
            "mod_w_profile='off' to silence this.",
            profile_name,
        )
        return model
    profile = MOD_W_PROFILES.get(profile_name) or MOD_W_PROFILES[DEFAULT_MOD_W_PROFILE]
    return _apply_mod_guidance(
        model,
        clip,
        positive,
        negative,
        None,
        quality_tags,
        quality_neg=quality_neg,
        w=profile["w"],
        start_layer=profile["start_layer"],
        end_layer=profile["end_layer"],
        taper=profile["taper"],
        taper_scale=profile["taper_scale"],
        final_w=profile["final_w"],
    )


def _run_spectrum(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise,
    *,
    smc_alpha=_SMC_CFG_ALPHA_DEFAULT,
    **extra,
):
    """spectrum_sample with the shared simple-node defaults (Spectrum presets,
    DCW off, fixed SMC λ). `extra` carries node-specific knobs (schedule,
    refresh_ratio, spd_*)."""
    return spectrum_sample(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        **_SPECTRUM_DEFAULTS,
        dcw_mode="off",
        smc_cfg_alpha=smc_alpha,
        smc_cfg_lambda=_SMC_CFG_LAMBDA_DEFAULT,
        **extra,
    )


# refresh_ratio: the single dial that selects + tunes SEA scheduling.
#   -1  → SEA off: plain growing-window schedule (accelerates from the first run,
#         no calibration pass — the old basic/mod-node behavior).
#    0  → SEA auto: match the window schedule's own refresh fraction at this step
#         count (compute-matched — same speed, just smarter step placement).
#   >0  → explicit SEA ratio (lower = faster, less faithful).
# Each distinct config calibrates + caches its own δ on first use.
_REFRESH_RATIO_INPUT = (
    "FLOAT",
    {
        "default": 0.0,
        "min": -1.0,
        "max": 1.0,
        "step": 0.01,
        "tooltip": (
            "SEA scheduling dial. -1 = SEA off (plain growing-window schedule; "
            "accelerates from the first run, no calibration). 0 = SEA auto (match "
            "the window schedule's refresh fraction at this step count — same "
            "speed, smarter step placement). >0 = explicit refresh ratio (lower = "
            "faster, less faithful). The first run at each (resolution / steps / "
            "cfg / refresh_ratio) does a one-time full-compute calibration pass, "
            "then caches δ to the ComfyUI user dir for later runs."
        ),
    },
)


def _schedule_for(refresh_ratio):
    """Map the refresh_ratio dial onto (schedule, refresh_ratio) for spectrum_sample."""
    if refresh_ratio < 0.0:
        return "window", -1.0
    return "sea", refresh_ratio


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class SpectrumKSampler:
    """Unified Spectrum sampler — acceleration + SEA scheduling + mod guidance.

    The one-stop drop-in. Chebyshev feature forecasting for ~2-3x speedup, the
    Spectral-Evolution-Aware (SEA) skip decision (``refresh_ratio`` dial; -1
    turns SEA off for the plain growing window), optional per-block modulation
    guidance toward quality tags (``mod_w_profile``; 'off' = no steering), and
    α-adaptive SMC-CFG for detail recovery. Subsumes the former
    SpectrumKSampler / SpectrumKSamplerModGuidance / SpectrumSEAKSamplerModGuidance
    nodes. DCW + full forecasting knobs live on the Advanced node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_KSAMPLER_INPUTS,
                **_MOD_PROFILE_INPUTS,
                "refresh_ratio": _REFRESH_RATIO_INPUT,
                "adaptive_smc_alpha": _SMC_CFG_ALPHA_INPUT,
                "fsg": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Foresight Guidance toward the golden path (one "
                            "switch = the validated production stack: CFG++ λ=1.5 "
                            "substrate + FSG band [0.59,0.75], K=3, on the 1024 "
                            "tier @ ~28 steps). Needs CFG != 1. Because CFG++ "
                            "replaces the cond/uncond combine, turning this ON "
                            "disables SMC-CFG (they are mutually exclusive). "
                            "Adds ~3·K forwards per in-band step. Band/K/Δσ/γ and "
                            "the CFG++ λ are tunable on the Advanced node; re-tune "
                            "if you change steps/resolution (the band moves)."
                        ),
                    },
                ),
            },
            "optional": {
                **_CLIP_INPUT,
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "Spectrum-accelerated sampler (drop-in KSampler replacement). Skips "
        "transformer blocks on predicted steps via Chebyshev feature "
        "forecasting for ~2-3x speedup. By default the SEA decision metric "
        "picks which steps to skip (refresh_ratio=0 → compute-matched to the "
        "growing window, just smarter; -1 → plain window, accelerates from run "
        "1). Optional modulation guidance steers toward quality tags via a "
        "learned pooled-text projection (wire a CLIP; the ~12MB adapter "
        "auto-downloads; set mod_w_profile='off' to skip). α-adaptive SMC-CFG "
        "recovers detail. DCW + full Spectrum tuning live on the Advanced node."
    )

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        quality_tags,
        mod_w_profile,
        refresh_ratio,
        fsg=False,
        quality_neg="",
        clip=None,
        denoise=1.0,
        adaptive_smc_alpha=_SMC_CFG_ALPHA_DEFAULT,
    ):
        m = _apply_mod_profile(
            model,
            clip,
            positive,
            negative,
            quality_tags,
            mod_w_profile,
            quality_neg=quality_neg,
        )
        schedule, refresh_ratio = _schedule_for(refresh_ratio)
        # The simple toggle enables the whole validated stack: CFG++ λ=1.5 +
        # FSG at the production band/K (spectrum_sample's defaults). CFG++
        # replaces the cond/uncond combine, so SMC-CFG must be off when FSG is on
        # (they both own sampler_cfg_function). Detail knobs live on Advanced.
        fsg_extra = (
            {"cfgpp_lambda": 1.5, "fsg_enabled": True} if fsg else {}
        )
        smc_alpha = 0.0 if fsg else adaptive_smc_alpha
        return _run_spectrum(
            m,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
            smc_alpha=smc_alpha,
            schedule=schedule,
            refresh_ratio=refresh_ratio,
            **fsg_extra,
        )


class SpectrumKSamplerAdvanced:
    """Full Spectrum sampler with modulation guidance and tunable forecasting parameters."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_KSAMPLER_INPUTS,
                **_mod_guidance_advanced_inputs(),
                **_SPECTRUM_INPUTS,
                **_DCW_INPUTS,
                **_FSG_INPUTS,
                "adaptive_smc_alpha": _SMC_CFG_ALPHA_INPUT,
                "smc_cfg_lambda": _SMC_CFG_LAMBDA_INPUT,
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "Spectrum-accelerated sampler with modulation guidance and full "
        "control over forecasting parameters. Combines quality steering "
        "via learned pooled-text projection with adjustable Chebyshev "
        "polynomial feature forecasting for tuned speed/quality tradeoff. "
        "Exposes DCW (post-step SNR-t bias correction, manual + auto modes) "
        "and α-adaptive SMC-CFG (velocity-space sliding-mode CFG combine)."
    )

    def sample(
        self,
        model,
        clip,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        adapter,
        quality_tags,
        mod_w,
        quality_neg="",
        mod_start_layer=8,
        mod_end_layer=27,
        mod_taper=0,
        mod_taper_scale=0.25,
        mod_final_w=0.0,
        denoise=1.0,
        window_size=2.0,
        flex_window=0.25,
        warmup_steps=6,
        blend_w=0.3,
        cheby_degree=3,
        ridge_lambda=0.1,
        dcw_mode="off",
        dcw_lambda=0.01,
        dcw_band_mask="LL",
        dcw_calibrator=AUTO_CALIBRATOR_SENTINEL,
        cfgpp_lambda=0.0,
        fsg=False,
        fsg_band_lo=0.59,
        fsg_band_hi=0.75,
        fsg_k=3,
        fsg_d_sigma=0.1,
        fsg_gamma=0.0,
        adaptive_smc_alpha=_SMC_CFG_ALPHA_DEFAULT,
        smc_cfg_lambda=_SMC_CFG_LAMBDA_DEFAULT,
    ):
        m = _apply_mod_guidance(
            model,
            clip,
            positive,
            negative,
            adapter,
            quality_tags,
            quality_neg=quality_neg,
            w=mod_w,
            start_layer=mod_start_layer,
            end_layer=mod_end_layer,
            taper=mod_taper,
            taper_scale=mod_taper_scale,
            final_w=mod_final_w,
        )
        return spectrum_sample(
            m,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            blend_w=blend_w,
            cheby_degree=cheby_degree,
            ridge_lambda=ridge_lambda,
            dcw_mode=dcw_mode,
            dcw_lambda=dcw_lambda,
            dcw_band_mask=dcw_band_mask,
            dcw_calibrator=dcw_calibrator,
            clip=clip,
            smc_cfg_alpha=adaptive_smc_alpha,
            smc_cfg_lambda=smc_cfg_lambda,
            cfgpp_lambda=cfgpp_lambda,
            fsg_enabled=fsg,
            fsg_band=(fsg_band_lo, fsg_band_hi),
            fsg_k=fsg_k,
            fsg_d_sigma=fsg_d_sigma,
            fsg_gamma=fsg_gamma,
        )


class AnimaModGuidance:
    """Standalone modulation-guidance model patcher.

    Pulls the quality-steering setup out of the sampler so it composes with any
    sampler (vanilla KSampler, Spectrum, or the SPEED node) by sitting upstream
    on the MODEL input. Returns a patched MODEL clone — the actual positive /
    negative conditioning is still wired into the sampler as usual; this node
    reads them only to compute the steering delta direction.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to patch with mod guidance."}),
                **_CLIP_INPUT,
                **_MOD_PROFILE_INPUTS,
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"
    DESCRIPTION = (
        "Modulation guidance as a standalone model patcher. Steers generation "
        "toward quality tags via a learned pooled-text projection into the AdaLN "
        "timestep embedding (the default ~12MB adapter is auto-downloaded on "
        "first use). Wire its output MODEL into any sampler — composes with the "
        "Spectrum and SPEED (Spectrum+SPD) samplers. Set mod_w_profile='off' to "
        "pass the model through unchanged."
    )

    def patch(
        self,
        model,
        clip,
        quality_tags,
        quality_neg,
        mod_w_profile,
        positive,
        negative,
    ):
        m = _apply_mod_profile(
            model,
            clip,
            positive,
            negative,
            quality_tags,
            mod_w_profile,
            quality_neg=quality_neg,
        )
        return (m,)


class DiTCFGFSGPatch:
    """Standalone correction patcher for DiT samplers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "DiT MODEL to patch with DCW, SMC-CFG, CFG++, and/or FSG."
                        )
                    },
                ),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Disable to pass the input MODEL through unchanged.",
                    },
                ),
                **_DCW_INPUTS,
                "smc_cfg": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable α-adaptive SMC-CFG. Mutually exclusive with "
                            "CFG++ because both replace sampler_cfg_function."
                        ),
                    },
                ),
                "adaptive_smc_alpha": _SMC_CFG_ALPHA_INPUT,
                "smc_cfg_lambda": _SMC_CFG_LAMBDA_INPUT,
                "cfgpp": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable CFG++ σ-scheduled CFG combine. Mutually "
                            "exclusive with SMC-CFG."
                        ),
                    },
                ),
                **_FSG_INPUTS,
                "replace_existing_cfg": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Allow this node to overwrite an existing sampler_cfg_function. "
                            "Leave false unless you intentionally want this patch to win."
                        ),
                    },
                ),
            },
            "optional": {
                "steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 10000,
                        "forceInput": True,
                        "tooltip": (
                            "Downstream sampler step count. Required when cfgpp or fsg is enabled."
                        ),
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "forceInput": True,
                        "tooltip": (
                            "Downstream sampler CFG scale. Required when smc_cfg, cfgpp, or fsg is enabled."
                        ),
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Downstream sampler name. Required when cfgpp or fsg is enabled."
                        ),
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Downstream scheduler name. Required when cfgpp or fsg is enabled."
                        ),
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "forceInput": True,
                        "tooltip": (
                            "Downstream sampler denoise value. Required when cfgpp or fsg is enabled."
                        ),
                    },
                ),
                "clip": (
                    "CLIP",
                    {
                        "tooltip": "CLIP encoder. Required when dcw_mode=auto.",
                    },
                ),
                "positive": (
                    "CONDITIONING",
                    {
                        "tooltip": (
                            "Positive conditioning. Required when dcw_mode=auto."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"
    DESCRIPTION = (
        "Standalone DiT correction model patch. Groups DCW, SMC-CFG, CFG++, and "
        "FSG in one patcher. Optional inputs are required only by enabled features: "
        "dcw_mode=auto needs clip and positive; smc_cfg needs cfg; cfgpp/fsg need "
        "steps, cfg, sampler_name, scheduler, and denoise."
    )
    OUTPUT_TOOLTIPS = (
        "MODEL clone patched with the selected correction hooks, or the input MODEL when disabled.",
    )

    def patch(
        self,
        model,
        enabled=True,
        dcw_mode="off",
        dcw_lambda=0.01,
        dcw_band_mask="LL",
        dcw_calibrator=AUTO_CALIBRATOR_SENTINEL,
        smc_cfg=False,
        adaptive_smc_alpha=_SMC_CFG_ALPHA_DEFAULT,
        smc_cfg_lambda=_SMC_CFG_LAMBDA_DEFAULT,
        cfgpp=False,
        cfgpp_lambda=0.0,
        fsg=False,
        fsg_band_lo=0.59,
        fsg_band_hi=0.75,
        fsg_k=3,
        fsg_d_sigma=0.1,
        fsg_gamma=0.0,
        replace_existing_cfg=False,
        steps=None,
        cfg=None,
        sampler_name=None,
        scheduler=None,
        denoise=None,
        clip=None,
        positive=None,
    ):
        if not enabled:
            return (model,)

        want_dcw = dcw_mode != "off" and (
            dcw_mode == "auto" or not math.isclose(float(dcw_lambda), 0.0)
        )
        want_smc = bool(smc_cfg) and float(adaptive_smc_alpha) > 0.0
        want_cfgpp = bool(cfgpp) and float(cfgpp_lambda) > 0.0
        want_fsg = bool(fsg) and int(fsg_k) > 0
        want_cfg_features = want_smc or want_cfgpp or want_fsg

        if not (want_dcw or want_cfg_features):
            return (model,)
        if want_smc and want_cfgpp:
            raise RuntimeError(
                "DiT CFG-FSG/DCW Patch: SMC-CFG and CFG++ both replace "
                "sampler_cfg_function. Disable one of them."
            )
        if want_smc and want_fsg:
            raise RuntimeError(
                "DiT CFG-FSG/DCW Patch: FSG is validated on CFG++/plain CFG, not on "
                "SMC-CFG. Disable either smc_cfg or fsg."
            )

        if dcw_mode == "auto":
            _require_option_values(
                "DiT CFG-FSG/DCW Patch dcw_mode=auto",
                hint=(
                    "Connect clip and positive optional inputs, or set dcw_mode "
                    "to manual/off."
                ),
                clip=clip,
                positive=positive,
            )

        cfg_value = None
        if want_cfg_features:
            _require_option_values(
                "DiT CFG-FSG/DCW Patch CFG correction",
                hint=(
                    "Connect cfg from the downstream sampler when smc_cfg, cfgpp, "
                    "or fsg is enabled."
                ),
                cfg=cfg,
            )
            cfg_value = float(cfg)
            if math.isclose(cfg_value, 1.0):
                logger.warning(
                    "DiT CFG-FSG/DCW Patch: CFG/FSG corrections need cfg != 1.0; ignoring them."
                )
                want_smc = want_cfgpp = want_fsg = False
                want_cfg_features = False

        if want_cfgpp or want_fsg:
            _require_option_values(
                "DiT CFG-FSG/DCW Patch CFG++/FSG",
                hint=(
                    "Connect steps, sampler_name, scheduler, and denoise from the "
                    "downstream sampler when cfgpp or fsg is enabled."
                ),
                steps=steps,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
            )
            steps = int(steps)
            sampler_name = str(sampler_name)
            scheduler = str(scheduler)
            denoise = float(denoise)

        if not (want_dcw or want_cfg_features):
            return (model,)
        if want_fsg and not want_cfgpp:
            logger.warning(
                "DiT CFG-FSG/DCW Patch: FSG is enabled without CFG++; this is an "
                "experimental plain-CFG substrate."
            )

        m = model.clone()
        _clone_model_options(m)

        if want_dcw:
            calibrator = None
            if dcw_mode == "auto":
                calibrator = setup_dcw_calibrator(m, clip, positive, dcw_calibrator)
            install_dcw(
                m,
                lam=float(dcw_lambda),
                schedule="one_minus_sigma",
                band_mask=dcw_band_mask,
                calibrator=calibrator,
            )

        if want_smc:
            _require_sampler_cfg_slot(m, "SMC-CFG", replace_existing_cfg)
            install_smc_cfg(
                m,
                alpha=float(adaptive_smc_alpha),
                lam=float(smc_cfg_lambda),
            )

        sigma_schedule = None
        if want_cfgpp or want_fsg:
            ks_sched = comfy.samplers.KSampler(
                m,
                steps=steps,
                device=m.load_device,
                sampler=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
            )
            sigma_schedule = [float(s) for s in ks_sched.sigmas]

        if want_cfgpp:
            _require_sampler_cfg_slot(m, "CFG++", replace_existing_cfg)
            install_cfgpp(m, lam=float(cfgpp_lambda), sigmas=sigma_schedule)

        if want_fsg:
            fsg_obj = FSGCalibrator(
                band=(float(fsg_band_lo), float(fsg_band_hi)),
                k=int(fsg_k),
                d_sigma=float(fsg_d_sigma),
                gamma=(float(fsg_gamma) if fsg_gamma and fsg_gamma > 0.0 else None),
            )
            fsg_steps = fsg_step_indices(fsg_obj, sigma_schedule, steps)
            install_fsg(m, fsg=fsg_obj, guidance_scale=cfg_value)
            _add_spectrum_cache_veto(m, _fsg_spectrum_cache_veto(fsg_steps))
            logger.info(
                "DiT CFG-FSG/DCW Patch: FSG active on %d in-band steps; Spectrum "
                "cached prediction is vetoed for those steps.",
                len(fsg_steps),
            )

        return (m,)


class DiTSpectrumPatch:
    """Standalone DiT Spectrum MODEL patcher.

    Applies only the original Spectrum final_layer-pre-feature forecasting path
    to a MODEL clone. The returned MODEL can be wired into ComfyUI's built-in
    KSampler, KSampler Advanced, Custom Sampler, or other sampler nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "DiT MODEL to patch with Spectrum."}),
                "steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Must match the downstream sampler's steps.",
                    },
                ),
                "window_size": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.25,
                        "tooltip": "Initial caching window; 1.0 disables cached steps.",
                    },
                ),
                "flex_window": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Window growth after each actual forward.",
                    },
                ),
                "warmup_steps": (
                    "INT",
                    {
                        "default": 6,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Initial steps forced to actual DiT forwards.",
                    },
                ),
                "tail_actual_steps": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Final steps forced to actual DiT forwards.",
                    },
                ),
                "blend_w": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Chebyshev/Taylor blend weight; 1.0 is pure Chebyshev.",
                    },
                ),
                "cheby_degree": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 10,
                        "tooltip": "Chebyshev polynomial degree.",
                    },
                ),
                "ridge_lambda": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.001,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Ridge regression regularization strength.",
                    },
                ),
                "history_size": (
                    "INT",
                    {
                        "default": 100,
                        "min": 5,
                        "max": 10000,
                        "tooltip": "Forecaster buffer size.",
                    },
                ),
                "enabled": ("BOOLEAN", {"default": True}),
                "one_sampler_only": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Apply Spectrum only to the first sampler run that "
                            "uses this patched MODEL within a workflow run; later "
                            "sampler runs (e.g. hi-res fix) pass through. Re-arms "
                            "on each new workflow execution."
                        ),
                    },
                ),
                "verbose": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Log actual/cached step decisions.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"
    DESCRIPTION = (
        "Standalone DiT Spectrum model patch. Forecasts the DiT feature before "
        "final_layer and re-runs only final_layer/unpatchify on cached steps. "
        "Wire before a normal KSampler, KSampler Advanced, or Custom Sampler. "
        "Does not perform sampling and does not include mod guidance, DCW, "
        "SMC-CFG, or SPEED/SPD."
    )

    def patch(
        self,
        model,
        steps=30,
        window_size=2.0,
        flex_window=0.25,
        warmup_steps=6,
        tail_actual_steps=3,
        blend_w=0.3,
        cheby_degree=3,
        ridge_lambda=0.1,
        history_size=100,
        enabled=True,
        one_sampler_only=False,
        verbose=False,
    ):
        patched = apply_dit_spectrum_patch(
            model,
            steps=steps,
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            tail_actual_steps=tail_actual_steps,
            blend_w=blend_w,
            cheby_degree=cheby_degree,
            ridge_lambda=ridge_lambda,
            history_size=history_size,
            enabled=enabled,
            one_sampler_only=one_sampler_only,
            verbose=verbose,
        )
        return (patched,)


class DiTSpectrumPatchAdvanced(DiTSpectrumPatch):
    """Standalone DiT Spectrum MODEL patcher with compatibility controls."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = DiTSpectrumPatch.INPUT_TYPES()
        inputs["required"]["compat_policy"] = _COMPAT_POLICY_INPUT
        return inputs

    DESCRIPTION = (
        "Standalone DiT Spectrum model patch with compatibility controls. "
        "Use 'legacy' for existing behavior, or 'conservative'/'strict' when "
        "multi-positive conditioning, chained model wrappers, Custom Sampler "
        "flows, or cache veto callbacks should fall back to actual DiT forwards "
        "instead of using uncertain cached predictions."
    )

    def patch(
        self,
        model,
        steps=30,
        window_size=2.0,
        flex_window=0.25,
        warmup_steps=6,
        tail_actual_steps=3,
        blend_w=0.3,
        cheby_degree=3,
        ridge_lambda=0.1,
        history_size=100,
        enabled=True,
        one_sampler_only=False,
        verbose=False,
        compat_policy="legacy",
    ):
        patched = apply_dit_spectrum_patch(
            model,
            steps=steps,
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            tail_actual_steps=tail_actual_steps,
            blend_w=blend_w,
            cheby_degree=cheby_degree,
            ridge_lambda=ridge_lambda,
            history_size=history_size,
            enabled=enabled,
            one_sampler_only=one_sampler_only,
            verbose=verbose,
            compat_policy=compat_policy,
        )
        return (patched,)


class SpectrumSPDKSampler:
    """KSampler (Spectrum + SPD) — the SPEED sampler.

    Low-res SPD prefix (uncached) → spectral expansion at the handoff →
    Spectrum-forecasted full-res tail. Consumes whatever MODEL it is given, so
    mod guidance composes by sitting upstream (AnimaModGuidance node).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_KSAMPLER_INPUTS,
                **_SPD_INPUTS,
                "adaptive_smc_alpha": _SMC_CFG_ALPHA_INPUT,
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "SPEED sampler: SPD multi-resolution progressive diffusion composed with "
        "Spectrum. Runs the low-res prefix (spd_scale) uncached, spectral-expands "
        "to full resolution at σ=spd_sigma, then Spectrum forecasts the full-res "
        "tail (phase-2-only naive-reset compose, validated at scale 0.5 / σ0.7). "
        "Stacks SPD's token saving on Spectrum's block saving. Euler-only "
        "(the σ schedule is re-spaced mid-loop). Mod guidance composes via the "
        "upstream AnimaModGuidance node. DCW lives on the Spectrum Advanced node."
    )

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        split_mode,
        spd_scale,
        spd_sigma,
        denoise=1.0,
        adaptive_smc_alpha=_SMC_CFG_ALPHA_DEFAULT,
    ):
        return _run_spectrum(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
            smc_alpha=adaptive_smc_alpha,
            spd_scale=spd_scale,
            spd_sigma=spd_sigma,
        )


class SpectrumSPDLoRAKSampler:
    """KSampler (SPD LoRA) — loads an SPD-trained LoRA and auto-schedules SPEED.

    A LoRA distilled by the SPD trajectory-adapter workflow (anima_lora
    ``make exp-spd``) is fit to a specific resolution schedule and snapshots it
    into its safetensors metadata. This node loads such a LoRA onto the MODEL and
    reads ``ss_spd_stages`` / ``ss_spd_transition_sigmas`` to drive the SPEED
    (SPD + Spectrum) sampler automatically — no manual scale/σ tuning, and the
    inference geometry matches what the adapter trained on. Chain a stock
    LoraLoader / AnimaModGuidance upstream to stack style LoRAs or mod guidance.
    """

    def __init__(self):
        self.loaded_lora = None  # (path, state_dict) cache, mirrors LoraLoader

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_KSAMPLER_INPUTS,
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {
                        "tooltip": (
                            "SPD-trained LoRA (anima_lora make exp-spd). Its "
                            "resolution schedule is read from the file's "
                            "ss_spd_stages / ss_spd_transition_sigmas metadata "
                            "and applied automatically."
                        ),
                    },
                ),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "LoRA weight multiplier applied to the MODEL.",
                    },
                ),
                "adaptive_smc_alpha": _SMC_CFG_ALPHA_INPUT,
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "Loads an SPD-trained LoRA and runs the SPEED (SPD + Spectrum) sampler "
        "with the resolution schedule auto-read from the adapter's metadata "
        "(ss_spd_stages / ss_spd_transition_sigmas — supports multi-stage). "
        "Aligns inference geometry with training, no manual scale/σ tuning. "
        "Euler-only (σ is re-spaced mid-loop). Falls back to the validated "
        "0.5 / σ0.7 single handoff only if the LoRA carries no schedule metadata. "
        "Mod guidance / extra LoRAs compose via upstream nodes; DCW lives on the "
        "Spectrum Advanced node."
    )

    def _apply_lora(self, model, lora_name, strength):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)
        # clip=None / strength_clip=0: model-only patch (LoraLoaderModelOnly form).
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
        return model_lora

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        lora_name,
        lora_strength,
        denoise=1.0,
        adaptive_smc_alpha=_SMC_CFG_ALPHA_DEFAULT,
    ):
        m = self._apply_lora(model, lora_name, lora_strength)

        spd_stages, spd_transition_sigmas, label = _read_spd_schedule_meta(lora_name)
        if spd_stages is None:
            logger.warning(
                "SPD LoRA '%s' has no ss_spd_stages metadata; falling back to the "
                "validated %.2f / σ%.2f single handoff.",
                lora_name,
                SPD_FALLBACK_SCALE,
                SPD_FALLBACK_SIGMA,
            )
        else:
            logger.info(
                "SPD LoRA '%s': auto schedule label=%s stages=%s σ=%s",
                lora_name,
                label,
                spd_stages,
                spd_transition_sigmas,
            )

        return _run_spectrum(
            m,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
            smc_alpha=adaptive_smc_alpha,
            spd_scale=SPD_FALLBACK_SCALE,
            spd_sigma=SPD_FALLBACK_SIGMA,
            spd_stages=spd_stages,
            spd_transition_sigmas=spd_transition_sigmas,
        )


NODE_CLASS_MAPPINGS = {
    "SpectrumKSampler": SpectrumKSampler,
    "SpectrumKSamplerAdvanced": SpectrumKSamplerAdvanced,
    "SpectrumSPDKSampler": SpectrumSPDKSampler,
    "SpectrumSPDLoRAKSampler": SpectrumSPDLoRAKSampler,
    "AnimaModGuidance": AnimaModGuidance,
    "DiTCFGFSGPatch": DiTCFGFSGPatch,
    "DiTSpectrumPatch": DiTSpectrumPatch,
    "DiTSpectrumPatchAdvanced": DiTSpectrumPatchAdvanced,
    # Deprecated aliases — the mod-guidance + SEA samplers are now folded into
    # the unified SpectrumKSampler (mod_w_profile + refresh_ratio dials). These
    # keys are kept so saved workflows referencing them still load; they are
    # intentionally absent from NODE_DISPLAY_NAME_MAPPINGS so they no longer
    # appear in the add-node menu.
    "SpectrumKSamplerModGuidance": SpectrumKSampler,
    "SpectrumSEAKSamplerModGuidance": SpectrumKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumKSampler": "KSampler (Spectrum)",
    "SpectrumKSamplerAdvanced": "KSampler (Spectrum + Mod Guidance Advanced)",
    "SpectrumSPDKSampler": "KSampler (Spectrum + SPD / SPEED)",
    "SpectrumSPDLoRAKSampler": "KSampler (SPD LoRA / auto-schedule)",
    "AnimaModGuidance": "Anima Mod Guidance (model patch)",
    "DiTCFGFSGPatch": "DiT CFG-FSG/DCW Patch",
    "DiTSpectrumPatch": "DiT Spectrum Patch",
    "DiTSpectrumPatchAdvanced": "DiT Spectrum Patch Advanced",
}
