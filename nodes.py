"""ComfyUI node definitions for Spectrum inference acceleration."""

import comfy.samplers
import folder_paths

from .mod_guidance import AUTO_ADAPTER_SENTINEL, setup_mod_guidance
from .spectrum import spectrum_sample

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
        "default": "absurdres, highres, masterpiece, best quality, score_9, score_8, newest, year 2025, year 2024",
        "multiline": True,
        "dynamicPrompts": True,
        "tooltip": "Quality tags to steer generation toward via modulation.",
    },
)

# Per-block guidance profiles. Each maps to a fixed (w, start, end, taper, taper_scale, final_w)
# tuple — see docs/mod-guidance.md for the naming + rationale.
# end_layer = -1 means "all blocks" (resolved against num_blocks at setup time).
MOD_W_PROFILES = {
    "step_i8_skip27": dict(w=3.0, start_layer=8,  end_layer=27, taper=0, taper_scale=0.25, final_w=0.0),
    "step_i14":       dict(w=3.0, start_layer=14, end_layer=-1, taper=0, taper_scale=0.25, final_w=0.0),
    "uniform_w3":     dict(w=3.0, start_layer=0,  end_layer=-1, taper=0, taper_scale=0.25, final_w=0.0),
}
DEFAULT_MOD_W_PROFILE = "step_i8_skip27"


_MOD_GUIDANCE_SIMPLE_INPUTS = {
    "clip": ("CLIP", {"tooltip": "CLIP encoder for encoding positive quality tags."}),
    "quality_tags": _QUALITY_TAGS_INPUT,
    "mod_w_profile": (
        list(MOD_W_PROFILES.keys()),
        {
            "default": DEFAULT_MOD_W_PROFILE,
            "tooltip": (
                "Per-block guidance schedule preset. "
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
        "clip": _MOD_GUIDANCE_SIMPLE_INPUTS["clip"],
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
                "default": -1,
                "min": -1,
                "max": 999,
                "tooltip": (
                    "Exclusive last block + 1. -1 = all remaining blocks. "
                    "Use 27 (Anima has 28 blocks) to skip the final compensation block."
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
            "default": 7,
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
    warmup_steps=7,
    blend_w=0.3,
    cheby_degree=3,
    ridge_lambda=0.1,
)

# DCW: SNR-t bias correction (arXiv:2604.16044). Anima form, λ < 0,
# schedule fixed to one_minus_sigma. See anima_lora/docs/methods/dcw.md.
_DCW_INPUTS = {
    "dcw_mode": (
        ["off", "manual", "auto"],
        {
            "default": "manual",
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


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class SpectrumKSampler:
    """Drop-in KSampler replacement with Spectrum acceleration using sensible defaults."""

    @classmethod
    def INPUT_TYPES(cls):
        # `clip` is required so auto-DCW can recover post-LLM-adapter c_pool.
        # Manual / off modes don't use it but the slot stays connected.
        return {
            "required": {
                **_KSAMPLER_INPUTS,
                "clip": ("CLIP", {"tooltip": "CLIP encoder (required for auto-DCW)."}),
                **_DCW_INPUTS,
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "Spectrum-accelerated sampler. Drop-in KSampler replacement that "
        "skips transformer blocks on predicted steps via Chebyshev polynomial "
        "feature forecasting for ~2-3x speedup. Uses sensible defaults."
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
        clip,
        denoise=1.0,
        dcw_mode="manual",
        dcw_lambda=0.01,
        dcw_band_mask="LL",
        dcw_calibrator=AUTO_CALIBRATOR_SENTINEL,
    ):
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
            dcw_mode=dcw_mode,
            dcw_lambda=dcw_lambda,
            dcw_band_mask=dcw_band_mask,
            dcw_calibrator=dcw_calibrator,
            clip=clip,
        )


class SpectrumKSamplerModGuidance:
    """Spectrum sampler with modulation guidance — quality steering via learned projection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {**_KSAMPLER_INPUTS, **_MOD_GUIDANCE_SIMPLE_INPUTS, **_DCW_INPUTS}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "Spectrum-accelerated sampler with modulation guidance. "
        "Steers generation toward quality tags via a learned pooled-text "
        "projection into the AdaLN timestep embedding. The default ~12MB "
        "pooled_text_proj adapter is auto-downloaded on first use. Quality "
        "tags are encoded through the full CLIP + LLM adapter pipeline for "
        "correct post-adapter pooling. Uses sensible Spectrum defaults and "
        "the 'step_i8' per-block guidance schedule (early-DC protected)."
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
        quality_tags,
        mod_w_profile,
        denoise=1.0,
        dcw_mode="manual",
        dcw_lambda=0.01,
        dcw_band_mask="LL",
        dcw_calibrator=AUTO_CALIBRATOR_SENTINEL,
    ):
        profile = MOD_W_PROFILES.get(mod_w_profile) or MOD_W_PROFILES[DEFAULT_MOD_W_PROFILE]
        m = model.clone()
        setup_mod_guidance(
            m,
            clip,
            positive,
            negative,
            None,
            quality_tags,
            profile["w"],
            start_layer=profile["start_layer"],
            end_layer=profile["end_layer"],
            taper=profile["taper"],
            taper_scale=profile["taper_scale"],
            final_w=profile["final_w"],
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
            **_SPECTRUM_DEFAULTS,
            dcw_mode=dcw_mode,
            dcw_lambda=dcw_lambda,
            dcw_band_mask=dcw_band_mask,
            dcw_calibrator=dcw_calibrator,
            clip=clip,
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
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "Spectrum-accelerated sampler with modulation guidance and full "
        "control over forecasting parameters. Combines quality steering "
        "via learned pooled-text projection with adjustable Chebyshev "
        "polynomial feature forecasting for tuned speed/quality tradeoff."
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
        mod_start_layer=8,
        mod_end_layer=-1,
        mod_taper=0,
        mod_taper_scale=0.25,
        mod_final_w=0.0,
        denoise=1.0,
        window_size=2.0,
        flex_window=0.25,
        warmup_steps=7,
        blend_w=0.3,
        cheby_degree=3,
        ridge_lambda=0.1,
        dcw_mode="manual",
        dcw_lambda=0.01,
        dcw_band_mask="LL",
        dcw_calibrator=AUTO_CALIBRATOR_SENTINEL,
    ):
        m = model.clone()
        setup_mod_guidance(
            m,
            clip,
            positive,
            negative,
            adapter,
            quality_tags,
            mod_w,
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
        )


NODE_CLASS_MAPPINGS = {
    "SpectrumKSampler": SpectrumKSampler,
    "SpectrumKSamplerModGuidance": SpectrumKSamplerModGuidance,
    "SpectrumKSamplerAdvanced": SpectrumKSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumKSampler": "KSampler (Spectrum)",
    "SpectrumKSamplerModGuidance": "KSampler (Spectrum + Mod Guidance)",
    "SpectrumKSamplerAdvanced": "KSampler (Spectrum + Mod Guidance Advanced)",
}
