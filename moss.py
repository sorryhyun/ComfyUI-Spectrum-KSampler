"""Spectrum KSampler for MOSS-Image, with optional per-block compile.

MOSS-Image (OpenMOSS) is a single-stream DiT loaded by the ``moss-image``
ComfyUI node. Its head is not Anima's ``final_layer`` / ``t_embedder`` /
``unpatchify``, so the DiT Spectrum patch cannot drive it:

* forecast target: the image tokens after the last block, i.e. the input of
  ``final_norm_video`` (``(1, H·W, dim)`` per sample);
* cached step: time embedding → ``video_head_modulation`` →
  ``final_norm_video`` → ``final_linear_video`` → ``(B, C, H, W)``, skipping the
  text refiner and every block (op-for-op with the model's own head);
* the model's forward loops over samples (each prompt at its true length), so
  the capture hook fires once per sample and the features are stacked in batch
  order.

Step scheduling, branch keys, one_sampler_only and cache vetoes are the DiT
patch's (``spectrum.py``). The forecaster is the linear form of
``SpectrumPredictor`` from anima_lora's ``library/moss/spectrum.py``: the fit
yields one weight per stored observation and the prediction is their weighted
sum, accumulated in fp32, instead of casting the whole history to fp32 on every
fit (+3 GiB of peak at 1024² with CFG there).

``compile`` folds the per-block ``torch.compile`` of ``diffusion_model.layers``
into the sampler (anima_lora ``library/moss/loader.py::compile_blocks``,
1.37x at 1024² with or without Spectrum).
"""

from __future__ import annotations

import logging

import torch

from .spectrum import (
    _advance_spectrum_state,
    _can_use_cached_prediction,
    _mark_spectrum_patch_consumed,
    _maybe_rearm_for_new_execution,
    _normalize_cond_or_uncond,
    _resolve_live_components,
    _spectrum_batch_keys,
)

from .feature_forecast import TokenSpectrumState, _update_forecasters

logger = logging.getLogger(__name__)

_MOSS_HEAD = (
    "layers",
    "time_embedding",
    "video_head_modulation",
    "final_norm_video",
    "final_linear_video",
    "freq_dim",
)


class MossSpectrumState(TokenSpectrumState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moss_feats = None


def _capture_pre_hook(module, args):
    state = getattr(module, "_spectrum_state", None)
    if state is not None and state.moss_feats is not None:
        state.moss_feats.append(args[0].detach().clone())


def _ensure_capture_hook(dit) -> None:
    norm = dit.final_norm_video
    if getattr(norm, "_spectrum_hook_installed", False):
        return
    norm.register_forward_pre_hook(_capture_pre_hook)
    norm._spectrum_hook_installed = True


def _require_moss(model):
    base = getattr(model, "model", None)
    dit = getattr(base, "diffusion_model", None)
    model_sampling = getattr(base, "model_sampling", None)
    missing = [n for n in _MOSS_HEAD if dit is None or not hasattr(dit, n)]
    if dit is None or missing or model_sampling is None:
        raise RuntimeError(
            "Spectrum for MOSS needs a MOSS-Image MODEL (moss-image loader); "
            f"diffusion_model is missing: {', '.join(missing) or 'model_sampling'}"
        )
    return dit, model_sampling


def _moss_head(dit, timestep: torch.Tensor, feat: torch.Tensor, height: int, width: int):
    """The model's head on forecast features; mirrors the tail of
    ``MOSSImageTransformer.forward`` with the batch done at once (the head is
    per-token)."""
    batch = feat.shape[0]
    times = timestep.to(feat.device, torch.float32).reshape(-1).expand(batch)
    half = dit.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    time = dit.time_embedding(embedding.to(dit.time_embedding[0].weight.dtype)).float()
    head = dit.video_head_modulation(time.to(dit.video_head_modulation.weight.dtype)).view(
        batch, 2, -1
    )
    x = dit.final_norm_video(feat.to(dit.final_norm_video.weight.dtype))
    x = x * (1 + head[:, 1, None]) + head[:, 0, None]
    x = dit.final_linear_video(x.to(dit.final_linear_video.weight.dtype))
    return x.transpose(1, 2).reshape(batch, -1, height, width)


def _cpp_compiler_available() -> bool:
    """Inductor's own compiler probe (``cl`` on Windows, g++/clang on POSIX)."""
    try:
        from torch._inductor.cpp_builder import get_cpp_compiler

        get_cpp_compiler()
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means compile won't work
        logger.warning(
            "Spectrum for MOSS: no working C++ compiler for inductor (%s); "
            "running the blocks eager.",
            exc,
        )
        return False


# The non-dynamic delegate of the last compiled checkpoint, keyed by its
# cached_patcher_init. clone(disable_dynamic=True) otherwise re-runs the loader
# on every sample (~3 s from disk at 1024²). One entry: it holds the DiT in host RAM.
_STATIC_BASE: dict = {}


def _static_base(init):
    try:
        key = (init[0], tuple(init[1]), init[2] if len(init) > 2 else None)
        hash(key)
    except TypeError:
        key = None
    base = _STATIC_BASE.get(key) if key is not None else None
    if base is None:
        base = init[0](*init[1], disable_dynamic=True)
        if len(init) > 2:
            base = base[init[2]]
        if key is not None:
            _STATIC_BASE.clear()
            _STATIC_BASE[key] = base
    return base


def _clone_for_compile(model):
    """A non-dynamic clone: under ComfyUI's dynamic VRAM loading the compiled
    blocks break once the DiT is re-staged (a new prompt loads the text encoder
    in between), failing with mismatched matmul shapes. The plain patcher shares
    one cached delegate model across samples, so only the first one loads it."""
    init = getattr(model, "cached_patcher_init", None)
    if init is not None and getattr(model, "is_dynamic", lambda: False)():
        base = _static_base(init)
        m = model.clone(disable_dynamic=True, model_override=base.get_clone_model_override())
        # Parent the clone on the patcher that owns its model. With the dynamic
        # input as parent, each sample's LoadedModel falls back to a patcher whose
        # .model is the dynamic one, so is_clone() never matches it again: one
        # stale entry per sample, all "memory leak with model MOSSImage" once
        # the workflow's loader chain is dropped while _STATIC_BASE keeps the DiT.
        m.parent = base
        return m
    try:
        return model.clone(disable_dynamic=True)
    except TypeError:
        logger.warning(
            "Spectrum for MOSS: this ComfyUI build can't disable dynamic VRAM "
            "loading via clone(); compile may fail under lazy weight loading."
        )
        return model.clone()


def _compile_layers(m) -> None:
    """``torch.compile`` each of ``diffusion_model.layers`` with dynamic shapes:
    the joint image+text length differs per prompt and per branch (as anima_lora's
    compile_blocks). A bounded ``mark_dynamic`` on the sequence axis only measured
    the same speed in ComfyUI (2026-10-17)."""
    from comfy_api.torch_helpers import set_torch_compile_wrapper

    num_layers = len(m.get_model_object("diffusion_model").layers)
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 16)
    set_torch_compile_wrapper(
        model=m,
        keys=[f"diffusion_model.layers.{i}" for i in range(num_layers)],
        backend="inductor",
        dynamic=True,
    )


def apply_moss_spectrum_patch(
    model,
    steps: int = 50,
    window_size: float = 2.0,
    flex_window: float = 0.5,
    warmup_steps: int = 6,
    tail_actual_steps: int = 3,
    blend_w: float = 0.3,
    cheby_degree: int = 3,
    ridge_lambda: float = 0.1,
    history_size: int = 100,
    enabled: bool = True,
    compile: bool = True,
    verbose: bool = False,
    one_sampler_only: bool = False,
):
    """Return a MODEL clone with Spectrum (when ``enabled``) and per-block
    compile (when ``compile``)."""
    # Before the early return: the sampler applies ModelSamplingSD3
    # (multiplier 1000) next, which is wrong for ComfyUI's built-in MOSS model.
    _require_moss(model)
    if not enabled and not compile:
        return model
    if history_size < cheby_degree + 2:
        raise RuntimeError(
            "Spectrum for MOSS requires history_size >= cheby_degree + 2 "
            f"(got history_size={history_size}, cheby_degree={cheby_degree})."
        )

    compile = compile and _cpp_compiler_available()
    m = _clone_for_compile(model) if compile else model.clone()
    dit, model_sampling = _require_moss(m)
    if compile:
        _compile_layers(m)
    if not enabled:
        return m

    state = MossSpectrumState(
        window_size=window_size,
        flex_window=flex_window,
        warmup_steps=warmup_steps,
        w=blend_w,
        m=cheby_degree,
        lam=ridge_lambda,
        num_steps=steps,
        tail_actual_steps=tail_actual_steps,
        history_size=history_size,
        verbose=verbose,
        one_sampler_only=one_sampler_only,
    )
    _ensure_capture_hook(dit)
    old_wrapper = m.model_options.get("model_function_wrapper")
    # A local, not `m`: a wrapper closing over the patcher forms a cycle through
    # m.model_options (see apply_dit_spectrum_patch).
    model_options = m.model_options

    def actual_forward(apply_model, args, input_x, timestep, c):
        if old_wrapper is not None:
            return old_wrapper(apply_model, args)
        return apply_model(input_x, timestep, **c)

    def moss_spectrum_wrapper(apply_model, args):
        input_x = args["input"]
        timestep = args["timestep"]
        c = args["c"]
        live_dit, live_model_sampling = _resolve_live_components(
            apply_model, dit, model_sampling, state
        )
        _maybe_rearm_for_new_execution(state)
        _ensure_capture_hook(live_dit)
        norm = live_dit.final_norm_video

        if state.patch_consumed:
            norm._spectrum_state = None
            return actual_forward(apply_model, args, input_x, timestep, c)
        new_step = _advance_spectrum_state(state, timestep.flatten()[0].item())
        if state.patch_consumed:
            norm._spectrum_state = None
            return actual_forward(apply_model, args, input_x, timestep, c)
        norm._spectrum_state = state

        cond_or_uncond, valid_chunks = _normalize_cond_or_uncond(args, input_x.shape[0])
        keys = _spectrum_batch_keys(c, cond_or_uncond)

        if new_step:
            state.mode = "cached" if valid_chunks and state.should_cache(keys) else "actual"
            if state.verbose:
                logger.info(
                    "Spectrum for MOSS: step %d/%d sigma=%.6g mode=%s",
                    state.step_idx + 1,
                    state.num_steps,
                    state.last_sigma,
                    state.mode,
                )

        if _can_use_cached_prediction(
            state, keys, input_x, timestep, c, old_wrapper, model_options, args, valid_chunks
        ):
            feat = torch.cat([state.forecasters[k].predict(float(state.step_idx)) for k in keys])
            t_internal = live_model_sampling.timestep(timestep).float()
            out = _moss_head(live_dit, t_internal, feat, input_x.shape[-2], input_x.shape[-1])
            _mark_spectrum_patch_consumed(state)
            return live_model_sampling.calculate_denoised(timestep, out.float(), input_x)

        state.mode = "actual"
        state.moss_feats = []
        try:
            result = actual_forward(apply_model, args, input_x, timestep, c)
            feats = state.moss_feats
        finally:
            state.moss_feats = None
        if state.active and valid_chunks and len(feats) == input_x.shape[0]:
            _update_forecasters(state, torch.cat(feats), input_x, keys)
        elif state.verbose:
            logger.warning(
                "Spectrum for MOSS: captured %d features for batch %d "
                "(cond_or_uncond=%s); no forecast update",
                len(feats),
                input_x.shape[0],
                cond_or_uncond,
            )
        _mark_spectrum_patch_consumed(state)
        return result

    m.set_model_unet_function_wrapper(moss_spectrum_wrapper)
    return m


def _float(default, lo, hi, step, tooltip):
    return ("FLOAT", {"default": default, "min": lo, "max": hi, "step": step, "tooltip": tooltip})


class MossSpectrumKSampler:
    """KSampler for MOSS-Image with the pipeline's sampling setup built in.

    ``pipeline_moss_image.py`` defaults: flow shift 5 (``ModelSamplingSD3``),
    CFG 3 with block 22 skipped on the uncond branch
    (``SkipLayerGuidanceDiTSimple``), 50 Euler steps on ``sgm_uniform``. Spectrum
    and per-block compile ride the same MODEL clone; sampling is ComfyUI's own
    ``common_ksampler``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MOSS-Image MODEL (moss-image loader)."}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True},
                ),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shift": _float(5.0, 0.0, 100.0, 0.01, "Flow shift (ModelSamplingSD3). MOSS pipeline: 5."),
                "uncond_skip_layers": (
                    "STRING",
                    {
                        "default": "22",
                        "tooltip": (
                            "Blocks skipped on the uncond branch (comma-separated; "
                            "empty = plain CFG). MOSS pipeline: 22."
                        ),
                    },
                ),
                "spectrum": ("BOOLEAN", {"default": True, "tooltip": "Spectrum feature forecasting."}),
                "flex_window": _float(
                    0.5,
                    0.0,
                    2.0,
                    0.05,
                    "Spectrum window growth after each actual forward. At 50 steps "
                    "0.25 / 0.5 / 1.0 run 22 / 19 / 16 actual forwards.",
                ),
                "compile": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "torch.compile each of the DiT's blocks (inductor, "
                            "dynamic shapes). Compiles on the first sample; falls "
                            "back to eager without a C++ compiler."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "KSampler for MOSS-Image: flow shift, uncond block-22 skip, Spectrum "
        "forecasting and per-block compile in one node. Replaces ModelSamplingSD3 "
        "+ SkipLayerGuidanceDiTSimple + KSampler."
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
        denoise=1.0,
        shift=5.0,
        uncond_skip_layers="22",
        spectrum=True,
        flex_window=0.5,
        compile=True,
    ):
        import nodes
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple

        m = apply_moss_spectrum_patch(
            model, steps=steps, flex_window=flex_window, enabled=spectrum, compile=compile
        )
        m = ModelSamplingSD3().patch(m, shift)[0]
        if uncond_skip_layers.strip():
            m = SkipLayerGuidanceDiTSimple.execute(
                m, 0.0, 1.0, double_layers="", single_layers=uncond_skip_layers
            ).result[0]
        return nodes.common_ksampler(
            m, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
            denoise=denoise,
        )
