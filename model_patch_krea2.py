"""Forecast-only adapter for ComfyUI's native Krea 2 MODEL.

The transformer and text fusion remain native on actual steps. Forecast steps
reuse the native timestep MLP and output head, including Comfy weight casting
and fixed LoRA patches. Reference images, temporal batches, input-token patches,
and activation-quantized output heads currently use actual forwards.
"""

import sys

from einops import rearrange


name = "Krea2"
key = "spectrum_native_krea2"


def require_model(model):
    from comfy.ldm.krea2.model import SingleStreamDiT

    base = model.model
    dit = getattr(base, "diffusion_model", None)
    sampling = getattr(base, "model_sampling", None)
    if not isinstance(dit, SingleStreamDiT) or sampling is None:
        raise RuntimeError("Spectrum for Krea2 requires a native ComfyUI Krea2 MODEL.")
    return dit, sampling


def _capture_pre_hook(module, args):
    from .model_patch import record_feature

    record_feature(module, args[0])


def capture_owner(dit):
    return dit.last


def capture(dit):
    owner = capture_owner(dit)
    return owner, owner.register_forward_pre_hook(_capture_pre_hook)


def extract_features(tensor, input_x, c):
    # txtfusion changes the feature dimension, but preserves the text length.
    text_length = c["c_crossattn"].shape[1]
    patch = 2  # Native Krea2 checkpoints use 2x2 latent patches.
    height, width = input_x.shape[-2:]
    image_tokens = ((height + patch - 1) // patch) * ((width + patch - 1) // patch)
    return tensor[:, text_length:text_length + image_tokens]


def capture_safe(dit, input_x, c):
    if input_x.ndim not in (4, 5) or (input_x.ndim == 5 and input_x.shape[2] != 1):
        return False
    if dit.patch != 2 or c.get("ref_latents"):
        return False
    options = c.get("transformer_options", {})
    if options.get("patches", {}).get("post_input"):
        return False
    linear = dit.last.linear
    if (
        getattr(linear, "layout_type", None) is not None
        and not getattr(linear, "_full_precision_mm", False)
        and not getattr(linear, "comfy_force_cast_weights", False)
        and not getattr(linear, "weight_function", ())
        and not getattr(linear, "bias_function", ())
    ):
        from comfy.quant_ops import QUANT_ALGOS

        # Native quantization sees text and image together. Slicing first can
        # change activation scales; current official INT8 Turbo has a BF16 head.
        if QUANT_ALGOS.get(getattr(linear, "quant_format", None), {}).get("quantize_input", True):
            return False
    return True


def head(dit, t_internal, feat, input_x):
    from comfy.ldm.flux.layers import timestep_embedding

    height, width = input_x.shape[-2:]
    patch = dit.patch
    # ModelSamplingFlux supplies sigma directly; timestep_embedding applies 1000.
    t = dit.tmlp(timestep_embedding(t_internal, dit.tdim).unsqueeze(1).to(feat.dtype))
    out = dit.last(feat, t)
    out = rearrange(
        out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=(height + patch - 1) // patch, w=(width + patch - 1) // patch,
        ph=patch, pw=patch, c=dit.channels,
    )
    out = out[:, :, :height, :width]
    if input_x.ndim == 5:
        out = out.unsqueeze(2)
    return out


def apply_krea2_model_patch(
    model,
    steps: int = 8,
    window_size: float = 2.0,
    flex_window: float = 0.25,
    warmup_steps: int = 3,
    tail_actual_steps: int = 2,
    blend_w: float = 0.3,
    cheby_degree: int = 1,
    ridge_lambda: float = 0.1,
    history_size: int = 8,
    enabled: bool = True,
    one_sampler_only: bool = False,
    verbose: bool = False,
    compat_policy: str = "conservative",
):
    from .model_patch import apply_spectrum_model_patch

    return apply_spectrum_model_patch(
        model, sys.modules[__name__], steps=steps, window_size=window_size,
        flex_window=flex_window, warmup_steps=warmup_steps,
        tail_actual_steps=tail_actual_steps, blend_w=blend_w,
        cheby_degree=cheby_degree, ridge_lambda=ridge_lambda,
        history_size=history_size, enabled=enabled,
        one_sampler_only=one_sampler_only, verbose=verbose,
        compat_policy=compat_policy,
    )
