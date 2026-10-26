"""MOSS MODEL patch for ComfyUI's built-in loader; custom-loader MOSS stays in moss.py."""
import sys

import torch

from .model_patch import apply_spectrum_model_patch, record_feature

name = "MOSS"
key = "spectrum_native_moss"


def require_model(model):
    # Older ComfyUI can still import and use the original custom-loader sampler.
    from comfy.ldm.moss_image.model import MOSSImage

    base = getattr(model, "model", None)
    dit = getattr(base, "diffusion_model", None)
    sampling = getattr(base, "model_sampling", None)
    if not isinstance(dit, MOSSImage) or sampling is None or not len(dit.layers):
        raise RuntimeError("Native MOSS Spectrum requires a native ComfyUI MOSS-Image MODEL.")
    return dit, sampling


def _capture_hook(module, args, output):
    record_feature(module, output)


def capture_owner(dit):
    # Native RMSNorm is fused: final_norm_video.forward itself is not called.
    return dit.layers[-1]


def capture(dit):
    owner = capture_owner(dit)
    return owner, owner.register_forward_hook(_capture_hook)


def extract_features(tensor, input_x, c):
    return tensor[:, :input_x.shape[-2] * input_x.shape[-1]]


def capture_safe(dit, input_x, c):
    replacements = c.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
    return ("single_block", len(dit.layers) - 1) not in replacements


def head(dit, timestep, feat, input_x):
    from comfy.ldm.moss_image.model import modulated_rms_norm, timestep_embedding

    batch = feat.shape[0]
    times = timestep.to(feat.device, torch.float32).reshape(-1).expand(batch)
    vec = dit.time_embedding(timestep_embedding(times, 256).to(feat.dtype))
    shift, scale = dit.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
    out = dit.final_linear_video(modulated_rms_norm(feat, dit.final_norm_video, scale, shift))
    return out.transpose(1, 2).reshape(batch, dit.latent_channels, *input_x.shape[-2:])


def apply_moss_model_patch(
    model, steps=30, window_size=2.0, flex_window=0.25, warmup_steps=6,
    tail_actual_steps=3, blend_w=0.3, cheby_degree=3, ridge_lambda=0.1,
    history_size=100, enabled=True, one_sampler_only=False, verbose=False,
    compat_policy="conservative",
):
    return apply_spectrum_model_patch(
        model, sys.modules[__name__], steps=steps, window_size=window_size,
        flex_window=flex_window, warmup_steps=warmup_steps, tail_actual_steps=tail_actual_steps,
        blend_w=blend_w, cheby_degree=cheby_degree, ridge_lambda=ridge_lambda,
        history_size=history_size, enabled=enabled, one_sampler_only=one_sampler_only,
        verbose=verbose, compat_policy=compat_policy,
    )
