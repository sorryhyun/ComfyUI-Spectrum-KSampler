"""DCW: post-step SNR-t bias correction (arXiv:2604.16044) for ComfyUI.

Anima form (pixel mode, opposite sign from paper, λ ≈ -0.015 for LL-only):

    diff           = x_in − last_denoised
    diff_masked    = haar_idwt(mask(haar_dwt(diff)))   # band restriction
    x_in          += λ · sched(σ_i) · diff_masked

Schedule defaults to ``one_minus_sigma`` (matches Anima's late-σ bias
envelope). Band mask defaults to ``LL`` — single-level Haar LL is a more
responsive lever than broadband and improves all four subbands; broadband
``all`` worsens detail bands. See ``anima_lora/docs/methods/dcw.md``.

Applied via two coordinated hooks installed on the model patcher:
  1) CALC_COND_BATCH wrapper — mutates ``x_in`` in-place at each new-step
     boundary using the previous step's cached post-CFG denoised. ``x_in``
     IS the sampler's ``x`` reference (passed all the way down from
     KSamplerX0Inpaint), so the Euler/ER-SDE/DPM step that follows
     operates on the corrected latent.
  2) sampler_post_cfg_function — captures the post-CFG denoised after
     each step for use by (1) on the next step.
"""

import logging
from typing import Optional

import torch

import comfy.patcher_extension

logger = logging.getLogger(__name__)

BANDS = ("LL", "LH", "HL", "HH")
ALL_BANDS = frozenset(BANDS)

_ODD_SHAPE_WARNED = False


def parse_band_mask(label: str) -> frozenset[str]:
    """CLI / widget string → frozenset of band names. ``all`` → all four bands.

    Format: ``LL``, ``HH``, ``LH+HL+HH``, ``all``. Case-insensitive on
    band names; ``all`` must be exactly that token.
    """
    if label == "all":
        return ALL_BANDS
    parts = [p.upper() for p in label.split("+") if p]
    bad = [p for p in parts if p not in BANDS]
    if bad or not parts:
        raise ValueError(
            f"unknown band(s) in mask {label!r}: {bad or '<empty>'}; "
            f"valid bands {BANDS!r} or 'all'"
        )
    return frozenset(parts)


def _haar_dwt_2d(
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-level 2D orthonormal Haar DWT on the last two dims.

    Returns (LL, LH, HL, HH), each (..., H/2, W/2). Requires even H, W.
    """
    a = v[..., 0::2, 0::2]
    b = v[..., 0::2, 1::2]
    c = v[..., 1::2, 0::2]
    d = v[..., 1::2, 1::2]
    s = 0.5
    LL = (a + b + c + d) * s
    LH = (a + b - c - d) * s
    HL = (a - b + c - d) * s
    HH = (a - b - c + d) * s
    return LL, LH, HL, HH


def _haar_idwt_2d(
    LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor
) -> torch.Tensor:
    s = 0.5
    a = (LL + LH + HL + HH) * s
    b = (LL + LH - HL - HH) * s
    c = (LL - LH + HL - HH) * s
    d = (LL - LH - HL + HH) * s
    out = torch.empty(
        *LL.shape[:-2], LL.shape[-2] * 2, LL.shape[-1] * 2,
        dtype=LL.dtype, device=LL.device,
    )
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out


def _band_masked_diff(
    diff: torch.Tensor, bands: frozenset[str]
) -> torch.Tensor:
    """DWT → zero non-mask bands → iDWT. Caller guarantees even H, W."""
    LL, LH, HL, HH = _haar_dwt_2d(diff)
    z = torch.zeros_like(LL)
    LL_m = LL if "LL" in bands else z
    LH_m = LH if "LH" in bands else z
    HL_m = HL if "HL" in bands else z
    HH_m = HH if "HH" in bands else z
    return _haar_idwt_2d(LL_m, LH_m, HL_m, HH_m)


class DCWState:
    def __init__(
        self,
        lam: float,
        schedule: str = "one_minus_sigma",
        bands: frozenset[str] = ALL_BANDS,
        calibrator=None,
    ):
        self.lam = lam
        self.schedule = schedule
        self.bands = bands
        # Auto mode: calibrator emits per-step λ (envelope already baked in,
        # so the wrapper applies it with schedule="const").
        self.calibrator = calibrator
        self.auto = calibrator is not None
        self.last_denoised: Optional[torch.Tensor] = None
        self.last_x_in: Optional[torch.Tensor] = None
        self.curr_sigma: Optional[float] = None
        # Integer step index — incremented on each new-σ boundary in the
        # CALC_COND_BATCH wrapper. Used by the calibrator's record / fire /
        # lambda_for_step.  -1 = no step has started yet.
        self.step_idx: int = -1

    def schedule_value(self, sigma_i: Optional[float]) -> float:
        if sigma_i is None:
            return 0.0
        if self.schedule == "one_minus_sigma":
            return 1.0 - sigma_i
        if self.schedule == "sigma_i":
            return sigma_i
        if self.schedule == "const":
            return 1.0
        return 0.0  # "none"


def _apply_correction_inplace(
    x_in: torch.Tensor, last_denoised: torch.Tensor, scalar: float, bands: frozenset[str]
) -> None:
    """In-place: x_in ← x_in + scalar · masked(x_in − last_denoised).

    When ``bands == ALL_BANDS``, falls through to the cheap fused
    ``add_`` (bit-identical to the broadband paper-form correction). For
    a band-restricted mask, computes the Haar DWT/iDWT round-trip in
    fp32 and casts back. If the latent has odd spatial dims, falls back
    to broadband with a one-time warning.
    """
    if bands == ALL_BANDS:
        x_in.add_(x_in - last_denoised, alpha=scalar)
        return

    H, W = x_in.shape[-2], x_in.shape[-1]
    if H % 2 or W % 2:
        global _ODD_SHAPE_WARNED
        if not _ODD_SHAPE_WARNED:
            logger.warning(
                "DCW band-mask requires even spatial dims (got HxW=%dx%d); "
                "falling back to broadband correction for this run.",
                H, W,
            )
            _ODD_SHAPE_WARNED = True
        x_in.add_(x_in - last_denoised, alpha=scalar)
        return

    diff = (x_in - last_denoised).float()
    masked = _band_masked_diff(diff, bands).to(x_in.dtype)
    x_in.add_(masked, alpha=scalar)


def _make_dcw_calc_cond_batch_wrapper(state: DCWState):
    def wrapper(executor, model, conds, x_in, timestep, model_options):
        # In flow-matching / CONST model_sampling, timestep == sigma.
        sigma = float(timestep[0]) if timestep.ndim else float(timestep)
        new_step = (
            state.curr_sigma is None or abs(sigma - state.curr_sigma) > 1e-8
        )
        if new_step:
            if state.last_denoised is not None:
                if state.auto:
                    # Calibrator returns the post-envelope λ_i directly.
                    # state.step_idx == previous step's index (we increment after).
                    s = state.calibrator.lambda_for_step(
                        state.step_idx, state.curr_sigma or 0.0
                    )
                elif state.lam != 0.0:
                    s = state.lam * state.schedule_value(state.curr_sigma)
                else:
                    s = 0.0
                if s != 0.0:
                    _apply_correction_inplace(
                        x_in, state.last_denoised, s, state.bands
                    )
            state.curr_sigma = sigma
            state.step_idx += 1
        # Cache x_in *after* correction; post-CFG hook needs it to recover the
        # post-CFG velocity v = (x_in - denoised) / σ for haar_LL_norm.
        if state.auto:
            state.last_x_in = x_in.detach()
        return executor(model, conds, x_in, timestep, model_options)

    return wrapper


def _make_dcw_post_cfg_hook(state: DCWState):
    def hook(args):
        # args["denoised"] is post-CFG x0_pred. Clone so the cache survives
        # downstream in-place ops on the sampler's tensors.
        denoised = args["denoised"]
        state.last_denoised = denoised.clone()
        if state.auto and state.calibrator is not None:
            calib = state.calibrator
            sigma = state.curr_sigma
            if (
                state.step_idx < calib.k_warmup
                and state.last_x_in is not None
                and sigma is not None
                and sigma > 0.0
            ):
                # Post-CFG velocity. CONST model_sampling: denoised = x_in - σ * v.
                v = (state.last_x_in - denoised) / sigma
                calib.record(state.step_idx, v)
            calib.fire_head_if_due(state.step_idx)
        return denoised

    return hook


def install_dcw(
    model_patcher,
    *,
    lam: float,
    schedule: str = "one_minus_sigma",
    band_mask: str = "LL",
    calibrator=None,
) -> None:
    """Register DCW hooks on a cloned ModelPatcher (no-op when disabled).

    Caller must clone the model patcher before passing it in — this
    mutates ``model_patcher.model_options`` and registers a post-CFG
    hook on the patcher.

    When ``calibrator`` is given, runs in auto mode: per-step λ comes from
    ``calibrator.lambda_for_step`` and ``band_mask`` is forced to ``LL`` to
    match the head's training distribution. When ``calibrator`` is ``None``,
    runs the legacy scalar-λ path; ``lam == 0`` makes that a no-op.
    """
    if calibrator is None and lam == 0.0:
        return
    bands = frozenset({"LL"}) if calibrator is not None else parse_band_mask(band_mask)
    state = DCWState(
        lam=lam, schedule=schedule, bands=bands, calibrator=calibrator
    )
    comfy.patcher_extension.add_wrapper(
        comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
        _make_dcw_calc_cond_batch_wrapper(state),
        model_patcher.model_options,
        is_model_options=True,
    )
    model_patcher.set_model_sampler_post_cfg_function(_make_dcw_post_cfg_hook(state))
