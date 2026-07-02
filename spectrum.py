"""Spectrum state management, fast-forward path, and shared sampling logic."""

from __future__ import annotations

import copy
import logging
import math
from typing import Dict, Hashable, List, Optional, Sequence

import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

try:
    # Per-execution prompt_id — lets one_sampler_only re-arm on each fresh
    # workflow run instead of staying consumed forever on a cached MODEL.
    from comfy_execution.utils import get_executing_context
except Exception:  # older ComfyUI without the execution-context contextvar
    get_executing_context = None

from networks.spectrum_forecast import SpectrumPredictor
from .spectrum_sea import l1rel, sea_filter
from .dcw import install_dcw
from .dcw_calibrator import setup_dcw_calibrator
from .smc_cfg import install_smc_cfg
from .fsg import FSGCalibrator, fsg_step_indices, install_cfgpp, install_fsg

logger = logging.getLogger(__name__)

COMPAT_POLICIES = ("legacy", "conservative", "strict")
DEFAULT_COMPAT_POLICY = "legacy"

# Anima DiT patches the latent with spatial_patch_size=2, so latent H/W must
# be even (equivalently pixel H/W mod-16). Users picking odd-mod-32 pixel
# sizes hit a PatchEmbed assertion deep inside the DiT — we pad bottom/right
# before sampling and crop back after.
_PATCH_MULTIPLE = 2
_ODD_LATENT_WARNED = False


def _pad_latent_to_patch_multiple(t: torch.Tensor, patch: int = _PATCH_MULTIPLE):
    """Replicate-pad bottom/right of a latent to the next multiple of ``patch``.

    Handles both 4-D ``(B, C, H, W)`` and 5-D video-DiT ``(B, C, T, H, W)`` latents.

    Returns ``(padded, (H, W))`` where (H, W) are the original spatial dims.
    The caller crops the sampled output back to (H, W) before returning to
    the user.  Replicate (vs zero) padding minimises edge artifacts from the
    bottom/right strip that is later cropped away.
    """
    H, W = t.shape[-2], t.shape[-1]
    pad_h = (-H) % patch
    pad_w = (-W) % patch
    if pad_h == 0 and pad_w == 0:
        return t, (H, W)
    global _ODD_LATENT_WARNED
    if not _ODD_LATENT_WARNED:
        logger.warning(
            "Spectrum: latent H,W=%d,%d is not divisible by %d "
            "(Anima DiT patch_size). Padding to %d,%d and cropping the output "
            "back. For exact framing, use pixel sizes that are multiples of %d.",
            H,
            W,
            patch,
            H + pad_h,
            W + pad_w,
            patch * 8,
        )
        _ODD_LATENT_WARNED = True
    # 5-D video-DiT latents (B, C, T, H, W) need a size-6 pad tuple under
    # "replicate"; pad W/H and leave the temporal (and leading) dims untouched.
    if t.ndim >= 5:
        pad = (0, pad_w, 0, pad_h, 0, 0)
    else:
        pad = (0, pad_w, 0, pad_h)
    return F.pad(t, pad, mode="replicate"), (H, W)


def _spectrum_fast_forward(
    dit, timestep: torch.Tensor, predicted_feature: torch.Tensor
) -> torch.Tensor:
    """Runs only t_embedder + final_layer + unpatchify on predicted features.

    Returns the same shape as diffusion_model.forward() — 5D for video DiTs.
    """
    if timestep.ndim == 1:
        timestep = timestep.unsqueeze(1)
    # The forecaster works in its own dtype (bf16) and the Taylor blend can
    # promote to fp32 via the captured feature, so the prediction dtype need
    # not match the model. Pin it to final_layer's weight dtype before re-entry
    # — otherwise fp16 models (e.g. `--fast fp16_accumulation`) raise
    # "mat1 and mat2 ... float != c10::Half". t_emb follows via the cast below.
    model_dtype = next(dit.final_layer.parameters()).dtype
    predicted_feature = predicted_feature.to(model_dtype)
    # Replicate the model's two-step t_embedder call: Timesteps (sinusoidal,
    # always float32) -> cast to model dtype -> TimestepEmbedding (linear layers).
    # Calling t_embedder as a single Sequential skips the intermediate cast.
    t_sinusoidal = dit.t_embedder[0](timestep)
    t_emb, adaln = dit.t_embedder[1](t_sinusoidal.to(predicted_feature.dtype))
    t_emb = dit.t_embedding_norm(t_emb)
    # Mod guidance: add cached pooled-text projection from the DIFFUSION_MODEL
    # wrapper.  On actual steps the wrapper computes base+delta from post-adapter
    # context and caches it on dit._mod_pooled_proj.  On cached steps we reuse
    # the last actual step's value (text doesn't change between steps).
    pooled_proj = getattr(dit, "_mod_pooled_proj", None)
    if pooled_proj is not None:
        pp = pooled_proj.unsqueeze(1).to(t_emb.dtype)
        if pp.shape[0] == t_emb.shape[0]:
            t_emb = t_emb + pp
        elif pp.shape[0] == 1:
            t_emb = t_emb + pp.expand_as(t_emb)
    x = dit.final_layer(predicted_feature, t_emb, adaln_lora_B_T_3D=adaln)
    return dit.unpatchify(x)


def _normalize_compat_policy(policy: Optional[str]) -> str:
    if policy in COMPAT_POLICIES:
        return policy
    logger.warning(
        "Spectrum: unknown compat_policy=%r; using %s",
        policy,
        DEFAULT_COMPAT_POLICY,
    )
    return DEFAULT_COMPAT_POLICY


def _spectrum_batch_keys(c, cond_or_uncond: Sequence[int]) -> list:
    """Return branch-stable forecaster keys for the current ComfyUI batch."""
    fallback = [int(cou) for cou in cond_or_uncond]
    transformer_options = c.get("transformer_options", {}) if isinstance(c, dict) else {}
    uuids = transformer_options.get("uuids")
    if uuids is None:
        return fallback
    if torch.is_tensor(uuids):
        uuids = uuids.detach().cpu().tolist()
    try:
        uuid_list = list(uuids)
    except TypeError:
        return fallback
    if len(uuid_list) != len(fallback):
        return fallback
    return [(cou, str(uid)) for cou, uid in zip(fallback, uuid_list)]


def _wrapper_cache_safe(old_wrapper) -> bool:
    if old_wrapper is None:
        return True
    if getattr(old_wrapper, "__spectrum_requires_actual__", False):
        return False
    return bool(getattr(old_wrapper, "__spectrum_cache_safe__", False))


def _uses_uuid_branch_keys(keys: Sequence[Hashable]) -> bool:
    return all(isinstance(key, tuple) and len(key) == 2 for key in keys)


def _spectrum_context_changed(
    state: SpectrumState, input_x: torch.Tensor, keys: Sequence[Hashable]
) -> bool:
    del keys
    if state.input_shape is not None and tuple(input_x.shape[1:]) != state.input_shape:
        return True
    return False


def _iter_cache_vetoes(model_options) -> list:
    if not isinstance(model_options, dict):
        return []
    vetoes = model_options.get("spectrum_cache_vetoes", [])
    if callable(vetoes):
        return [vetoes]
    try:
        return [v for v in vetoes if callable(v)]
    except TypeError:
        return []


def _passes_cache_vetoes(
    state: SpectrumState,
    args,
    keys: Sequence[Hashable],
    input_x: torch.Tensor,
    timestep: torch.Tensor,
    c,
    model_options,
) -> bool:
    for veto in _iter_cache_vetoes(model_options):
        try:
            allowed = veto(
                state=state,
                args=args,
                keys=keys,
                input_x=input_x,
                timestep=timestep,
                c=c,
            )
        except Exception as e:
            logger.warning(
                "Spectrum: cache veto callback %r failed (%s); using actual forward",
                veto,
                e,
            )
            return False
        if allowed is False:
            return False
    return True


def _can_use_cached_prediction(
    state: SpectrumState,
    keys: Sequence[Hashable],
    input_x: torch.Tensor,
    timestep: torch.Tensor,
    c,
    old_wrapper,
    model_options,
    args,
    valid_chunks: bool,
) -> bool:
    if state.mode != "cached" or not valid_chunks or not state.has_forecasters(keys):
        return False
    if not _passes_cache_vetoes(
        state, args, keys, input_x, timestep, c, model_options
    ):
        return False
    if state.compat_policy == "legacy":
        return True
    if not _wrapper_cache_safe(old_wrapper):
        if state.verbose:
            logger.info(
                "Spectrum: compat_policy=%s blocks cached step because the "
                "previous model_function_wrapper is not cache-safe",
                state.compat_policy,
            )
        return False
    if state.compat_policy == "strict" and not _uses_uuid_branch_keys(keys):
        if state.verbose:
            logger.info(
                "Spectrum: compat_policy=strict blocks cached step because "
                "conditioning UUID branch keys are unavailable"
            )
        return False
    if state.step_idx >= state.num_steps:
        if state.verbose:
            logger.info(
                "Spectrum: compat_policy=%s blocks cached step after expected "
                "steps were exceeded (%d >= %d)",
                state.compat_policy,
                state.step_idx,
                state.num_steps,
            )
        return False
    if _spectrum_context_changed(state, input_x, keys):
        if state.verbose:
            logger.info(
                "Spectrum: compat_policy=%s blocks cached step after latent "
                "shape change",
                state.compat_policy,
            )
        return False
    return True


def _update_forecasters_from_feature(
    state: SpectrumState,
    feat: Optional[torch.Tensor],
    input_x: torch.Tensor,
    keys: Sequence[Hashable],
    valid_chunks: bool,
    label: str,
) -> None:
    if not state.active or feat is None or not valid_chunks:
        return
    batch_chunks = len(keys)
    if batch_chunks == 0 or feat.shape[0] % batch_chunks != 0:
        if state.verbose:
            logger.warning(
                "%s: feature batch %d cannot be split into %d conditioning chunks; "
                "skipping forecast update",
                label,
                feat.shape[0],
                batch_chunks,
            )
        return

    feat_chunks = feat.chunk(batch_chunks, dim=0)
    if len(feat_chunks) != batch_chunks:
        if state.verbose:
            logger.warning(
                "%s: feature batch %d produced %d chunks for %d conditioning keys; "
                "skipping forecast update",
                label,
                feat.shape[0],
                len(feat_chunks),
                batch_chunks,
            )
        return

    def _create_or_update() -> None:
        for idx, key in enumerate(keys):
            if key not in state.forecasters:
                state.forecasters[key] = SpectrumPredictor(
                    state.m_param,
                    state.lam,
                    state.w,
                    feat.device,
                    feat_chunks[idx].shape,
                    state.num_steps,
                    K=state.history_size,
                )
            state.forecasters[key].update(float(state.step_idx), feat_chunks[idx])

    try:
        _create_or_update()
    except AssertionError:
        if state.verbose:
            logger.info("%s: feature shape changed; resetting forecasters", label)
        state.clear_forecasters()
        _create_or_update()
    state.record_context(input_x, keys)


class SpectrumState:
    def __init__(
        self,
        window_size: float,
        flex_window: float,
        warmup_steps: int,
        w: float,
        m: int,
        lam: float,
        num_steps: int,
        tail_actual_steps: int = 3,
        history_size: int = 100,
        verbose: bool = False,
        one_sampler_only: bool = False,
        schedule: str = "window",
        refresh_ratio: float = -1.0,
        sea_beta: float = 2.0,
        delta: Optional[float] = None,
        fsg_steps: Optional[frozenset] = None,
        compat_policy: str = DEFAULT_COMPAT_POLICY,
    ):
        self.window_size = window_size
        self.flex_window = flex_window
        self.warmup_steps = warmup_steps
        self.w = w
        self.m_param = m
        self.lam = lam
        self.num_steps = num_steps
        self.tail_actual_steps = tail_actual_steps
        self.history_size = history_size
        self.verbose = verbose
        self.one_sampler_only = one_sampler_only
        self.compat_policy = _normalize_compat_policy(compat_policy)

        # SEA schedule (SeaCache decision metric). schedule="sea" replaces the
        # growing-window rule with accumulate-until-δ on the SEA-filtered latent.
        # delta is None while uncalibrated → the loop falls back to the window
        # rule and records the per-step distance trace for one-shot auto-δ.
        self.schedule = schedule
        self.refresh_ratio = refresh_ratio
        self.sea_beta = sea_beta
        self.delta = delta
        self.sea_accum = 0.0
        self.sea_prev: Optional[torch.Tensor] = None
        self.sea_dists: List[float] = []  # decision-region trace, calibration only

        # FSG: step indices forced to actual forwards (the latent is calibrated
        # before these steps, so a cached feature forecast would be stale) and
        # excluded from the SEA decision denominator — same treatment as
        # warmup/tail. Empty when FSG is off.
        self.fsg_steps: frozenset = fsg_steps or frozenset()

        # Runtime
        self.step_idx = -1
        self.last_sigma: Optional[float] = None
        self.mode = "actual"
        self.curr_ws = window_size
        self.consec_cached = 0
        self.fwd_count = 0
        self.steps_seen = 0  # cumulative forwards across SPD resets (logging)

        # When False, every step runs an actual forward and no forecaster is
        # built — used by the SPEED sampler to keep the low-res SPD prefix
        # uncached (phase-2-only). Flipped True (with a reset) at the handoff.
        self.active = True

        # Forecasters keyed by conditioning branch. Legacy paths can still use
        # cond_or_uncond ints, but modern ComfyUI supplies per-conditioning UUIDs.
        self.forecasters: Dict[Hashable, SpectrumPredictor] = {}
        self.captured_feat: Optional[torch.Tensor] = None
        self.patch_consumed = False
        self.input_shape: Optional[tuple] = None

        # one_sampler_only: the ComfyUI prompt_id this state was last armed for.
        # The patched MODEL (and this state) is cached across workflow re-runs,
        # so we re-arm when the prompt_id changes — otherwise patch_consumed
        # would stay True forever and Spectrum would no-op on every later queue.
        self.active_prompt_id = None

    def reset(self) -> None:
        """Re-arm a fresh warmup window, discarding the current forecasters.

        Called by the SPEED sampler at the SPD resolution handoff: the captured
        ``final_layer`` feature changes token grid across the transition, so the
        stage-0 forecasters are unusable and Spectrum must re-warm on the
        full-res tail. ``fwd_count`` / ``steps_seen`` are left intact so the
        end-of-sample speedup log spans both phases.
        """
        self.step_idx = -1
        self.last_sigma = None
        self.mode = "actual"
        self.curr_ws = self.window_size
        self.consec_cached = 0
        self.clear_forecasters()
        self.captured_feat = None
        self.sea_accum = 0.0
        self.sea_prev = None
        self.sea_dists = []

    def clear_forecasters(self) -> None:
        self.forecasters = {}
        self.input_shape = None

    def record_context(self, input_x: torch.Tensor, keys: Sequence[Hashable]) -> None:
        del keys
        self.input_shape = tuple(input_x.shape[1:])

    def observe_sea(self, latent: torch.Tensor, sigma: float) -> None:
        """Accrue the SEA-filtered latent distance for the current step.

        Called once per new sampler step (after step_idx advanced, before the
        cache decision) on the input latent x_t. Under CFG the batch tiles the
        same x_t across cond/uncond, so row 0 is x_t. Mirrors the training-repo
        loop: distance accrues into ``sea_accum`` (reset on each refresh, Eq. 8)
        and, during the uncalibrated pass only, the raw per-step distance is
        recorded over the decision region for one-shot auto-δ.
        """
        if self.schedule != "sea":
            return
        x = latent[0:1]  # (1, C, H, W) — x_t
        sea_now = sea_filter(x, float(sigma), self.sea_beta)
        if self.sea_prev is not None:
            d = l1rel(sea_now, self.sea_prev)
            self.sea_accum += d
            stop_at = self.num_steps - self.tail_actual_steps
            if (
                self.delta is None
                and self.warmup_steps <= self.step_idx < stop_at
                and self.step_idx not in self.fsg_steps
            ):
                self.sea_dists.append(d)
        self.sea_prev = sea_now

    def _forecaster_ready(self, key: Hashable) -> bool:
        forecaster = self.forecasters.get(key)
        if forecaster is None:
            return False
        return forecaster.cheb.t_buf.numel() >= max(2, self.m_param + 2)

    def forecasters_ready(self, keys: Sequence[Hashable]) -> bool:
        return all(self._forecaster_ready(key) for key in keys)

    def should_cache(self, keys: Optional[Sequence[Hashable]] = None) -> bool:
        if not self.active:
            return False
        if self.step_idx < self.warmup_steps:
            return False
        stop_at = self.num_steps - self.tail_actual_steps
        if self.step_idx >= stop_at:
            return False
        if self.step_idx in self.fsg_steps:
            return False  # FSG-calibrated step — must run an actual forward
        if keys is not None and not self.forecasters_ready(keys):
            return False
        if self.schedule == "sea" and self.delta is not None:
            # Refresh (actual) once the accumulated SEA distance crosses δ; cache
            # (skip) until then. The accumulator resets on each refresh in the
            # step-advance bookkeeping (alongside consec_cached).
            return self.sea_accum < self.delta
        # Window schedule, or SEA calibration pass (δ uncalibrated) → window rule.
        return (self.consec_cached + 1) % max(1, math.floor(self.curr_ws)) != 0

    def has_forecasters(self, keys: Sequence[Hashable]) -> bool:
        return all(key in self.forecasters for key in keys)


def _capture_pre_hook(module, args):
    """Module-singleton pre-hook on final_layer — stores the pre-final feature
    on whichever SpectrumState is currently bound to the module.
    """
    state = getattr(module, "_spectrum_state", None)
    if state is not None:
        state.captured_feat = args[0].detach().clone()


def _ensure_capture_hook(dit) -> None:
    final_layer = dit.final_layer
    if getattr(final_layer, "_spectrum_hook_installed", False):
        return
    final_layer.register_forward_pre_hook(_capture_pre_hook)
    final_layer._spectrum_hook_installed = True


def _resolve_live_components(apply_model, fallback_dit, fallback_model_sampling, state):
    """Resolve the DiT + model_sampling that actually run *this* forward.

    The wrapper is invoked as ``model_function_wrapper(model.apply_model, ...)``,
    so ``apply_model.__self__`` is the live BaseModel. ComfyUI can hand the
    sampler a *different* DiT instance than the one patched at ``apply_dit_..``
    time — most commonly a downstream ``AnimaBlockCompile`` clones the model with
    ``disable_dynamic=True``, which rebuilds ``diffusion_model``. The patch-time
    refs then point at a dead module: the capture hook never fires, forecasters
    never fill, and Spectrum silently runs every step actual (looks like the
    forecaster keeps resetting). Prefer the live module and fall back only if it
    can't be resolved. Mirrors mod_guidance's re-home.
    """
    owner = getattr(apply_model, "__self__", None)
    dit = getattr(owner, "diffusion_model", None) if owner is not None else None
    model_sampling = (
        getattr(owner, "model_sampling", None) if owner is not None else None
    )
    if dit is None:
        dit = fallback_dit
    elif dit is not fallback_dit and not getattr(state, "_rehomed_logged", False):
        state._rehomed_logged = True
        if state.verbose:
            logger.info(
                "DiT Spectrum Patch: re-homing to live diffusion_model "
                "(patch-time id=%x != live id=%x); reinstalling capture hook.",
                id(fallback_dit) & 0xFFFFFF,
                id(dit) & 0xFFFFFF,
            )
    if model_sampling is None:
        model_sampling = fallback_model_sampling
    return dit, model_sampling


def _require_dit_spectrum_components(model):
    missing = []
    base = getattr(model, "model", None)
    dit = getattr(base, "diffusion_model", None)
    model_sampling = getattr(base, "model_sampling", None)

    if dit is None:
        missing.append("model.model.diffusion_model")
    else:
        for name in (
            "final_layer",
            "t_embedder",
            "t_embedding_norm",
            "unpatchify",
        ):
            if not hasattr(dit, name):
                missing.append(f"model.model.diffusion_model.{name}")
    if model_sampling is None:
        missing.append("model.model.model_sampling")

    if missing:
        raise RuntimeError(
            "DiT Spectrum Patch requires a DiT-style model with these "
            f"components: {', '.join(missing)}"
        )
    return dit, model_sampling


def _clone_model_options(model):
    try:
        model.model_options = copy.deepcopy(model.model_options)
    except Exception as e:
        logger.warning(
            "DiT Spectrum Patch: deepcopy(model_options) failed (%s); using a "
            "shallow copy for wrapper isolation.",
            e,
        )
        model.model_options = dict(model.model_options)


def _normalize_cond_or_uncond(args, batch_size: int):
    raw = args.get("cond_or_uncond", [0])
    if raw is None:
        raw = [0]
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().tolist()
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    cond_or_uncond = [int(cou) for cou in raw]
    if len(cond_or_uncond) == 0:
        cond_or_uncond = [0]
    if batch_size % len(cond_or_uncond) != 0:
        return cond_or_uncond, False
    return cond_or_uncond, True


def _advance_spectrum_state(state: SpectrumState, sigma_val: float) -> bool:
    """Advance state once per new sampler sigma/timestep.

    Returns True when a new sampling step was observed. If sigma rises, assume a
    fresh sampler run started on the same patched MODEL and reset the forecast
    buffers before counting the new step.
    """
    eps = 1e-8
    if state.last_sigma is not None and sigma_val > state.last_sigma + eps:
        if state.one_sampler_only and state.steps_seen > 0:
            state.patch_consumed = True
            if state.verbose:
                logger.info(
                    "DiT Spectrum Patch: detected another sampler run; "
                    "one_sampler_only is passing through"
                )
            return False
        if state.verbose:
            logger.info(
                "DiT Spectrum Patch: sigma increased %.6g -> %.6g; resetting state",
                state.last_sigma,
                sigma_val,
            )
        state.reset()

    if state.last_sigma is not None and abs(sigma_val - state.last_sigma) <= eps:
        return False

    if state.step_idx >= 0:
        if state.mode == "actual":
            state.fwd_count += 1
            if state.step_idx >= state.warmup_steps:
                state.curr_ws = round(state.curr_ws + state.flex_window, 3)
            state.consec_cached = 0
        else:
            state.consec_cached += 1

    state.step_idx += 1
    state.steps_seen += 1
    state.last_sigma = sigma_val
    return True


def _maybe_rearm_for_new_execution(state: SpectrumState) -> None:
    """Re-arm a ``one_sampler_only`` patch at the start of each workflow run.

    ComfyUI caches the patched MODEL output, so the same ``SpectrumState`` is
    reused across re-queues. Once consumed, ``patch_consumed`` would otherwise
    stay True for the life of the process and Spectrum would silently no-op on
    every subsequent run. We key the arming on the current execution's
    ``prompt_id``: a *different* prompt_id means a fresh workflow run → re-arm;
    the *same* prompt_id (a later sampler in the same graph, e.g. hi-res fix)
    keeps the consumed state, preserving the intended one-sampler-only behavior.
    """
    if not state.one_sampler_only or get_executing_context is None:
        return
    ctx = get_executing_context()
    prompt_id = getattr(ctx, "prompt_id", None)
    if prompt_id is None or prompt_id == state.active_prompt_id:
        return
    state.active_prompt_id = prompt_id
    if state.patch_consumed or state.steps_seen > 0:
        state.reset()
    state.patch_consumed = False
    state.steps_seen = 0
    state.fwd_count = 0


def _mark_spectrum_patch_consumed(state: SpectrumState) -> None:
    if not state.one_sampler_only or state.patch_consumed:
        return
    if state.step_idx >= max(0, state.num_steps - 1):
        state.patch_consumed = True
        if state.verbose:
            logger.info(
                "DiT Spectrum Patch: one_sampler_only consumed after %d steps",
                state.step_idx + 1,
            )


def apply_dit_spectrum_patch(
    model,
    steps: int = 30,
    window_size: float = 2.0,
    flex_window: float = 0.25,
    warmup_steps: int = 6,
    tail_actual_steps: int = 3,
    blend_w: float = 0.3,
    cheby_degree: int = 3,
    ridge_lambda: float = 0.1,
    history_size: int = 100,
    enabled: bool = True,
    verbose: bool = False,
    one_sampler_only: bool = False,
    compat_policy: str = DEFAULT_COMPAT_POLICY,
):
    """Return a MODEL clone patched with DiT Spectrum feature forecasting only."""
    if not enabled:
        return model
    if history_size < cheby_degree + 2:
        raise RuntimeError(
            "DiT Spectrum Patch requires history_size >= cheby_degree + 2 "
            f"(got history_size={history_size}, cheby_degree={cheby_degree})."
        )

    m = model.clone()
    _clone_model_options(m)
    dit, model_sampling = _require_dit_spectrum_components(m)

    state = SpectrumState(
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
        compat_policy=compat_policy,
    )

    _ensure_capture_hook(dit)
    old_wrapper = m.model_options.get("model_function_wrapper")

    def actual_forward(apply_model, args, input_x, timestep, c):
        if old_wrapper is not None:
            return old_wrapper(apply_model, args)
        return apply_model(input_x, timestep, **c)

    def passthrough_forward(apply_model, args, input_x, timestep, c, live_dit):
        live_dit.final_layer._spectrum_state = None
        return actual_forward(apply_model, args, input_x, timestep, c)

    def spectrum_model_patch_wrapper(apply_model, args):
        input_x = args["input"]
        timestep = args["timestep"]
        c = args["c"]

        # Resolve the DiT/model_sampling that actually run this forward — a
        # downstream compile node may have rebuilt diffusion_model, stranding
        # the patch-time refs (see _resolve_live_components).
        live_dit, live_model_sampling = _resolve_live_components(
            apply_model, dit, model_sampling, state
        )

        # Re-arm one_sampler_only when ComfyUI re-runs the workflow (new
        # prompt_id); the cached MODEL would otherwise stay consumed forever.
        _maybe_rearm_for_new_execution(state)

        if state.patch_consumed:
            return passthrough_forward(
                apply_model, args, input_x, timestep, c, live_dit
            )

        # Re-home the capture hook + state onto the live final_layer every call,
        # so a reused *or rebuilt* diffusion_model never writes into a stale
        # (or dead) patch state.
        _ensure_capture_hook(live_dit)
        live_dit.final_layer._spectrum_state = state

        cond_or_uncond, valid_chunks = _normalize_cond_or_uncond(args, input_x.shape[0])
        keys = _spectrum_batch_keys(c, cond_or_uncond)
        if state.compat_policy != "legacy" and _spectrum_context_changed(
            state, input_x, keys
        ):
            state.clear_forecasters()
        sigma_val = timestep.flatten()[0].item()
        new_step = _advance_spectrum_state(state, sigma_val)
        if state.patch_consumed:
            return passthrough_forward(
                apply_model, args, input_x, timestep, c, live_dit
            )
        if new_step:
            state.mode = (
                "cached"
                if valid_chunks and state.should_cache(keys)
                else "actual"
            )
            if state.verbose:
                logger.info(
                    "DiT Spectrum Patch: step %d/%d sigma=%.6g mode=%s",
                    state.step_idx + 1,
                    state.num_steps,
                    sigma_val,
                    state.mode,
                )

        if _can_use_cached_prediction(
            state,
            keys,
            input_x,
            timestep,
            c,
            old_wrapper,
            m.model_options,
            args,
            valid_chunks,
        ):
            predictions = []
            for key in keys:
                predictions.append(
                    state.forecasters[key].predict(float(state.step_idx))
                )

            batched_feat = torch.cat(predictions, dim=0)
            t_internal = live_model_sampling.timestep(timestep).to(batched_feat.dtype)
            noise_pred = _spectrum_fast_forward(live_dit, t_internal, batched_feat)
            result = live_model_sampling.calculate_denoised(
                timestep, noise_pred.float(), input_x
            )
            _mark_spectrum_patch_consumed(state)
            return result

        state.mode = "actual"
        state.captured_feat = None

        result = actual_forward(apply_model, args, input_x, timestep, c)

        feat = state.captured_feat
        _update_forecasters_from_feature(
            state, feat, input_x, keys, valid_chunks, "DiT Spectrum Patch"
        )
        if state.verbose and not valid_chunks:
            logger.warning(
                "DiT Spectrum Patch: cond_or_uncond=%s does not divide batch=%d; "
                "running actual forward without forecast update",
                cond_or_uncond,
                input_x.shape[0],
            )

        _mark_spectrum_patch_consumed(state)
        return result

    m.set_model_unet_function_wrapper(spectrum_model_patch_wrapper)
    return m


def spectrum_sample(
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
    window_size,
    flex_window,
    warmup_steps,
    blend_w,
    cheby_degree,
    ridge_lambda,
    dcw_mode: str = "manual",
    dcw_lambda: float = 0.01,
    dcw_schedule: str = "one_minus_sigma",
    dcw_band_mask: str = "LL",
    dcw_calibrator: Optional[str] = None,
    clip=None,
    smc_cfg_alpha: float = 0.0,
    smc_cfg_lambda: float = 5.0,
    spd_scale: float = 1.0,
    spd_sigma: float = 1.0,
    spd_stages=None,
    spd_transition_sigmas=None,
    schedule: str = "window",
    refresh_ratio: float = -1.0,
    sea_beta: float = 2.0,
    cfgpp_lambda: float = 0.0,
    fsg_enabled: bool = False,
    fsg_band=(0.59, 0.75),
    fsg_k: int = 3,
    fsg_d_sigma: float = 0.1,
    fsg_gamma: float = 0.0,
    compat_policy: str = DEFAULT_COMPAT_POLICY,
):
    """Shared Spectrum sampling logic used by all node tiers.

    spd_scale / spd_sigma: legacy single-handoff SPD (SPEED) multi-resolution
        knobs. When ``spd_scale < 1`` and ``0 < spd_sigma < 1`` the denoise loop
        is driven by the custom SPEED sampler (see ``spd.make_speed_sampler``):
        the ``spd_scale`` low-res prefix runs uncached, then at ``σ ≤ spd_sigma``
        the latent is spectral-expanded to full resolution and Spectrum forecasts
        the tail (phase-2-only naive-reset compose; ``bench/spd/compose_report.md``).
        Forces Euler. Defaults (1.0, 1.0) = no SPD, vanilla Spectrum path.
    spd_stages / spd_transition_sigmas: explicit multi-stage schedule (lists),
        e.g. ``[0.5, 0.75, 1.0]`` / ``[0.7, 0.4]``. When given they take
        precedence over the scalars above — this is how the LoRA-SPD node feeds a
        schedule read from an SPD-trained adapter's ``ss_spd_*`` metadata. See
        ``spd.resolve_spd_schedule``.

    dcw_mode: "off" / "manual" / "auto".
        - off: no DCW correction.
        - manual: scalar λ × schedule(σ_i), tunable via dcw_lambda + dcw_band_mask.
            Default 0.01 is the verified hyperparam for CFG ≥ ~2 with LL-only.
        - auto: per-step λ predicted by an OnlineDCWCalibrator fusion head.
            Requires ``clip`` (the same CLIP encoder feeding ``positive``) to
            recover post-LLM-adapter c_pool. ``dcw_calibrator`` names the
            artifact (or the auto-download sentinel). Forces band_mask = LL.

    dcw_lambda: scalar DCW strength used in manual mode. 0.0 = no-op even
        if mode != off. See anima_lora/docs/methods/dcw.md.
    dcw_band_mask: Subband restriction (manual mode only). Default 'LL' is
        strictly better than broadband on Anima.

    smc_cfg_alpha: α-adaptive Sliding-Mode Control CFG gain. ``0`` disables
        the modified CFG combine entirely (vanilla CFG path). ``0.2`` is the
        production default — α=0.2 puts the bang-bang correction at ~20% of
        the per-step mean residual magnitude, recovering detail without
        injecting visible chattering. Velocity-space combine (preserves
        across-step correctness when σ varies). Requires CFG ≠ 1 (auto-skipped).
    smc_cfg_lambda: SMC sliding-manifold slope λ. Paper sweep {3,4,5,6}; 5 best.

    cfgpp_lambda: CFG++ substrate strength λ (0 = off). Replaces the constant-w
        cond/uncond combine with the σ-scheduled CFG++ weight (paper App A.2);
        the substrate faithful FSG is defined on. λ=1.5 is the production point
        (tracks CFG=4 saturation/contrast). Mutually exclusive with SMC-CFG.
    fsg_enabled: Foresight Guidance pre-step latent calibration toward the
        golden path. Runs K forward-backward fixed-point iterations on the latent
        before each in-band step (forced to actual Spectrum forwards). Needs
        CFG ≠ 1. Pairs with cfgpp_lambda=1.5 for the production fsg/cfg++ point.
    fsg_band / fsg_k / fsg_d_sigma / fsg_gamma: FSG knobs. Band (σ_lo, σ_hi) is
        where calibration fires — default [0.59, 0.75] is the 1024-tier/28-step
        er_sde point; it moves DOWN for more steps and low-token (~768px) renders,
        UP for fewer steps (re-tune if you change steps/resolution; σ≈0.94 always
        diverges). K=3 iterations (each ~3 extra forwards). Δσ=0.1 stride.
        fsg_gamma=0 → use the CFG scale (=guidance); keep ≈4 even under CFG++
        (matching it to the CFG++ effective weight diverges).
    """
    compat_policy = _normalize_compat_policy(compat_policy)
    m = model.clone()

    # SMC-CFG: replace the CFG combine before any sampler call. Alpha=0 is
    # the universal off-switch; CFG=1 also short-circuits since there is no
    # cond/uncond residual to slide on.
    if smc_cfg_alpha > 0.0 and not math.isclose(cfg, 1.0):
        has_external_cfg = m.model_options.get("sampler_cfg_function") is not None
        if compat_policy != "legacy" and has_external_cfg:
            logger.warning(
                "Spectrum: compat_policy=%s found an existing sampler_cfg_function; "
                "skipping SMC-CFG so it is not overwritten.",
                compat_policy,
            )
        else:
            install_smc_cfg(m, alpha=smc_cfg_alpha, lam=smc_cfg_lambda)

    # Auto mode: load + setup the calibrator. If anything fails, fall back to
    # manual semantics (dcw_lambda × schedule) — never hard-error mid-sample.
    calibrator = None
    if dcw_mode == "auto":
        if warmup_steps < 7:
            raise RuntimeError(
                f"auto-DCW needs spectrum warmup_steps >= calibrator k_warmup (=7); "
                f"got warmup_steps={warmup_steps}. Use manual mode or raise warmup."
            )
        if positive is None or clip is None:
            logger.warning(
                "auto-DCW: missing clip / positive — falling back to manual."
            )
        else:
            calibrator = setup_dcw_calibrator(m, clip, positive, dcw_calibrator)

    # DCW: register CALC_COND_BATCH wrapper + post-CFG hook.
    if dcw_mode == "off":
        pass  # no hooks
    else:
        install_dcw(
            m,
            lam=dcw_lambda,
            schedule=dcw_schedule,
            band_mask=dcw_band_mask,
            calibrator=calibrator,
        )

    # CFG++ substrate + FSG foresight calibration. Both need CFG (a cond/uncond
    # gap) and the σ schedule (CFG++ maps σ_i → σ_next for its reweight; FSG
    # forces its in-band steps to actual forwards). CFG++ is mutually exclusive
    # with SMC-CFG (both own sampler_cfg_function). The σ schedule is recomputed
    # the way comfy.sample.sample will inside the loop, so the indices/weights
    # line up. SPD re-spaces σ mid-loop, so neither composes with it.
    do_cfg = not math.isclose(cfg, 1.0)
    smc_active = smc_cfg_alpha > 0.0 and do_cfg
    fsg = None
    fsg_steps: frozenset = frozenset()
    want_cfgpp = cfgpp_lambda and cfgpp_lambda > 0.0
    spd_will_own_loop = bool(spd_stages) or (
        0.0 < spd_scale < 1.0 and 0.0 < spd_sigma < 1.0
    )
    if (want_cfgpp or fsg_enabled) and not do_cfg:
        logger.warning("CFG++/FSG need CFG (cfg != 1.0); ignoring.")
        want_cfgpp = fsg_enabled = False
    if (want_cfgpp or fsg_enabled) and spd_will_own_loop:
        logger.warning("CFG++/FSG are not wired into SPD/SPEED; ignoring.")
        want_cfgpp = fsg_enabled = False

    if want_cfgpp or fsg_enabled:
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
            if smc_active:
                logger.warning(
                    "CFG++ and SMC-CFG both replace the cond/uncond combine; "
                    "ignoring CFG++ (SMC-CFG is active)."
                )
            else:
                install_cfgpp(m, lam=float(cfgpp_lambda), sigmas=sigma_schedule)
                logger.info("CFG++ substrate active (λ=%.3g).", cfgpp_lambda)

        if fsg_enabled:
            fsg = FSGCalibrator(
                band=tuple(fsg_band),
                k=int(fsg_k),
                d_sigma=float(fsg_d_sigma),
                gamma=(float(fsg_gamma) if fsg_gamma and fsg_gamma > 0.0 else None),
            )
            fsg_steps = fsg_step_indices(fsg, sigma_schedule, steps)
            install_fsg(m, fsg=fsg, guidance_scale=cfg)
            logger.info(
                "FSG active: band=[%.2f, %.2f], K=%d, Δσ=%.3g, %d in-band steps "
                "(+~%d fwd).",
                fsg.band[0],
                fsg.band[1],
                fsg.k,
                fsg.d_sigma,
                len(fsg_steps),
                3 * fsg.k * len(fsg_steps),
            )

    # SEA schedule: resolve the auto-δ target + load any cached δ for this config.
    # An uncached config runs one window-scheduled calibration pass (full compute)
    # then persists δ; later generates at the same config use the SEA trigger.
    # Mirror SpectrumState's tail default; threaded to both the cache key's
    # stop_at and the state so the decision region can never drift between them.
    tail_actual_steps = 3
    sea_key = None
    sea_delta = None
    if schedule == "sea" and (spd_stages or spd_scale < 1.0):
        logger.warning(
            "Spectrum SEA is incompatible with SPD/SPEED (mid-loop σ re-spacing "
            "breaks the distance trace); falling back to the window schedule."
        )
        schedule = "window"
    if schedule == "sea":
        from . import spectrum_sea as _sea

        stop_at = steps - tail_actual_steps
        if refresh_ratio <= 0.0:
            refresh_ratio = _sea.window_decision_fraction(
                steps, warmup_steps, stop_at, window_size, flex_window
            )
        h_lat, w_lat = (
            int(latent_image["samples"].shape[-2]),
            int(latent_image["samples"].shape[-1]),
        )
        # CFG++ λ and the FSG forced-step set move the trajectory δ is
        # calibrated against, so fold them into the key — a plain run and an
        # fsg/cfg++ run at the same geometry must never share a cached δ.
        sea_extra = ""
        if want_cfgpp and not smc_active:
            sea_extra += f"cfgpp{round(float(cfgpp_lambda), 4)}"
        if fsg is not None:
            sea_extra += f"fsg{sorted(fsg_steps)}k{fsg.k}d{round(fsg.d_sigma, 3)}"
        sea_key = _sea.make_cache_key(
            steps,
            warmup_steps,
            stop_at,
            refresh_ratio,
            cfg,
            sampler_name,
            h_lat,
            w_lat,
            extra=sea_extra,
        )
        sea_delta = _sea.load_delta(sea_key)
        logger.info(
            "Spectrum SEA: refresh_ratio=%.3f, δ=%s (%s)",
            refresh_ratio,
            f"{sea_delta:.4g}" if sea_delta is not None else "uncalibrated",
            "cached → SEA trigger"
            if sea_delta is not None
            else "calibrating this run (window schedule, full compute)",
        )

    state = SpectrumState(
        window_size=window_size,
        flex_window=flex_window,
        warmup_steps=warmup_steps,
        w=blend_w,
        m=cheby_degree,
        lam=ridge_lambda,
        num_steps=steps,
        tail_actual_steps=tail_actual_steps,
        schedule=schedule,
        refresh_ratio=refresh_ratio,
        sea_beta=sea_beta,
        delta=sea_delta,
        fsg_steps=fsg_steps,
        compat_policy=compat_policy,
    )

    dit = m.model.diffusion_model
    model_sampling = m.model.model_sampling

    # Install capture hook once per DiT instance (no-op on subsequent runs) and
    # bind this sample's state to the module. The hook reads state from the
    # module attribute, so its identity/closure is stable across samples —
    # torch.compile's dynamo cache survives between runs.
    _ensure_capture_hook(dit)
    dit.final_layer._spectrum_state = state

    old_wrapper = m.model_options.get("model_function_wrapper")

    def spectrum_wrapper(apply_model, args):
        input_x = args["input"]
        timestep = args["timestep"]
        c = args["c"]
        cond_or_uncond, valid_chunks = _normalize_cond_or_uncond(
            args, input_x.shape[0]
        )
        keys = _spectrum_batch_keys(c, cond_or_uncond)
        if state.compat_policy != "legacy" and _spectrum_context_changed(
            state, input_x, keys
        ):
            state.clear_forecasters()

        sigma_val = timestep.flatten()[0].item()

        if state.last_sigma is None or abs(sigma_val - state.last_sigma) > 1e-8:
            if state.step_idx >= 0:
                if state.mode == "actual":
                    state.fwd_count += 1
                    if state.step_idx >= state.warmup_steps:
                        state.curr_ws = round(state.curr_ws + state.flex_window, 3)
                    state.consec_cached = 0
                    state.sea_accum = 0.0  # refresh resets the SEA accumulator (Eq. 8)
                else:
                    state.consec_cached += 1

            state.step_idx += 1
            state.steps_seen += 1
            state.last_sigma = sigma_val
            # Accrue the SEA distance on this step's x_t before the cache decision.
            state.observe_sea(input_x, sigma_val)
            state.mode = (
                "cached" if valid_chunks and state.should_cache(keys) else "actual"
            )

        if _can_use_cached_prediction(
            state,
            keys,
            input_x,
            timestep,
            c,
            old_wrapper,
            m.model_options,
            args,
            valid_chunks,
        ):
            predictions = []
            for key in keys:
                pred_feat = state.forecasters[key].predict(float(state.step_idx))
                predictions.append(pred_feat)

            batched_feat = torch.cat(predictions, dim=0)
            t_internal = model_sampling.timestep(timestep).to(batched_feat.dtype)
            noise_pred = _spectrum_fast_forward(dit, t_internal, batched_feat)
            return model_sampling.calculate_denoised(
                timestep, noise_pred.float(), input_x
            )

        state.mode = "actual"
        state.captured_feat = None

        if old_wrapper is not None:
            result = old_wrapper(apply_model, args)
        else:
            result = apply_model(input_x, timestep, **c)

        feat = state.captured_feat
        _update_forecasters_from_feature(
            state, feat, input_x, keys, valid_chunks, "Spectrum"
        )
        if state.verbose and not valid_chunks:
            logger.warning(
                "Spectrum: cond_or_uncond=%s does not divide batch=%d; running "
                "actual forward without forecast update",
                cond_or_uncond,
                input_x.shape[0],
            )

        return result

    m.set_model_unet_function_wrapper(spectrum_wrapper)

    latent_img = latent_image["samples"].clone()
    latent_img = comfy.sample.fix_empty_latent_channels(
        m, latent_img, latent_image.get("downscale_ratio_spacial")
    )

    # Pad to mod-2 latent (Anima DiT patch_size=2) before noise / sampling so
    # odd-shape latents (mod-8 pixel but not mod-16) don't trip PatchEmbed.
    latent_img, orig_hw = _pad_latent_to_patch_multiple(latent_img)
    pad_h = latent_img.shape[-2] - orig_hw[0]
    pad_w = latent_img.shape[-1] - orig_hw[1]

    batch_inds = latent_image.get("batch_index")
    noise = comfy.sample.prepare_noise(latent_img, seed, batch_inds)

    noise_mask = latent_image.get("noise_mask")
    if noise_mask is not None and (pad_h or pad_w):
        # Pad with ones so the appended strip denoises normally; we crop it off.
        noise_mask = F.pad(noise_mask, (0, pad_w, 0, pad_h), mode="constant", value=1.0)
    callback = latent_preview.prepare_callback(m, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    # SPD (SPEED) takes over the loop when a real low→full transition is asked
    # for. It must own the loop (mid-loop resolution change + σ re-space), so it
    # runs through a custom KSAMPLER via sample_custom rather than the string
    # sampler path. Everything upstream (SMC / DCW / mod-guidance / the Spectrum
    # wrapper + capture hook) is already installed on ``m`` and composes.
    from .spd import make_speed_sampler, resolve_spd_schedule

    spd_stages_r, spd_trans_r, spd_active = resolve_spd_schedule(
        spd_stages, spd_transition_sigmas, spd_scale, spd_sigma
    )
    if spd_active and sampler_name != "euler":
        logger.warning(
            "SPEED/SPD re-spaces σ mid-loop and is Euler-only; ignoring requested "
            "sampler '%s' and using Euler.",
            sampler_name,
        )

    try:
        if spd_active:
            # Phase-2-only: the SPEED sampler flips state.active True at the handoff.
            state.active = False
            ks = comfy.samplers.KSampler(
                m,
                steps=steps,
                device=m.load_device,
                sampler=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                model_options=m.model_options,
            )
            sampler_obj = make_speed_sampler(state, spd_stages_r, spd_trans_r, seed)
            samples = comfy.sample.sample_custom(
                m,
                noise,
                cfg,
                sampler_obj,
                ks.sigmas,
                positive,
                negative,
                latent_img,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )
        else:
            samples = comfy.sample.sample(
                m,
                noise,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_img,
                denoise=denoise,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )
    finally:
        dit.final_layer._spectrum_state = None
        if hasattr(dit, "_mod_pooled_proj"):
            del dit._mod_pooled_proj

    if pad_h or pad_w:
        samples = samples[..., : orig_hw[0], : orig_hw[1]].contiguous()

    if state.step_idx >= 0:
        if state.mode == "actual":
            state.fwd_count += 1
        else:
            state.consec_cached += 1

    actual = state.fwd_count
    # SPD resets step_idx at the handoff, so step_idx only spans the tail; use
    # the cumulative step counter for the across-phase total. Note the low-res
    # prefix forwards are cheaper than full-res, so this block-skip ratio
    # understates the true SPEED wall-clock speedup.
    total = state.steps_seen if spd_active else state.step_idx + 1
    speedup = total / max(1, actual)
    do_cfg = not math.isclose(cfg, 1.0)
    cfg_note = " (x2 for CFG)" if do_cfg else ""
    tag = "SPEED (SPD+Spectrum)" if spd_active else "Spectrum"
    logger.info(
        f"{tag}: {actual}/{total} actual forwards "
        f"({speedup:.2f}x block-skip ratio{cfg_note})"
    )

    # SEA auto-δ: this generate ran the window schedule while recording the SEA
    # distance trace — solve the δ that hits the target refresh fraction and cache
    # it so subsequent generates at this config use the SEA trigger.
    if (
        schedule == "sea"
        and sea_delta is None
        and sea_key is not None
        and state.sea_dists
    ):
        from . import spectrum_sea as _sea

        new_delta = _sea.solve_delta_for_refresh_ratio(state.sea_dists, refresh_ratio)
        _sea.save_delta(sea_key, new_delta)
        logger.info(
            "Spectrum SEA: auto-calibrated δ=%.4g (target refresh_ratio=%.3f over "
            "%d decision steps); cached → subsequent generates at this config use "
            "the SEA trigger.",
            new_delta,
            refresh_ratio,
            len(state.sea_dists),
        )

    out = latent_image.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return (out,)
