"""CNS — Colored Noise Sampling, shipped as an ``er_sde_cns`` sampler entry.

Training-free SDE plug-in (Davidson, Issachar, Benaim — arXiv 2605.30332,
Algorithm 1). Replaces the **white** noise that the ER-SDE solver injects each
step with **frequency-colored**, RMS-normalized noise that dumps the fixed
stochastic-energy budget into the radial frequency bands the network has *not
yet resolved* at that step (per a precomputed completion matrix γ(f, t)). It is
a zero-sum reallocation of a fixed variance budget — not a global noise
scale-up — so RMS renormalization (paper §A) is load-bearing.

CNS lives at the sampler's per-step noise-injection seam, which no model patch
can reach (the injected noise is drawn *inside* the sampler, downstream of every
``model_options`` hook — same reason SPD owns its loop, see ``spd.py``). So we
ship it as a custom ER-SDE sample function and register it under the name
``er_sde_cns`` in ComfyUI's global sampler dropdown: pick it in the
``sampler_name`` field of any KSampler and the per-step noise is recolored, full
strength, with the shipped Anima γ matrix.

The recolorer (``CNSRecolorer``) and ``radial_bins`` are vendored verbatim from
``anima_lora/library/inference/corrections/cns.py``; ``sample_er_sde_cns`` is a
copy of ``comfy.k_diffusion.sampling.sample_er_sde`` with the single
noise-injection line routed through the recolorer (``float(sigmas[i])`` σ, to
match the CLI's ``ERSDESampler._sample_noise`` bit-for-bit). With CNS disabled
the math is identical to stock ``er_sde``.

Notes:
  * **Anima- and MOSS-calibrated.** The bundled Anima γ is measured on
    Anima's spectral-bias staircase at cfg=4. A MOSS-Image model (detected from
    the live diffusion_model) switches to a bundled MOSS γ — Base or SFT by a
    weight fingerprint, at the calibrated cfg (3/4/5) nearest the guider's — and
    recolors on the un-patchified grid. Any other model gets the Anima γ, mis-calibrated (still
    variance-conserving, so degraded-not-broken).
  * **er_sde only.** This is the stochastic ER-SDE solver; there is no ODE/euler
    surface for colored noise (white draw → no-op). Composes with the Spectrum
    model wrapper and the DCW / SMC-CFG / mod-guidance model patches (different
    seams), exactly as the CLI stacks ``--cns`` with ``SPECTRUM=1`` / ``MOD=1``.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np
import torch

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
# Anima γ, bundled (anima_lora scripts/calibration/cns_calibrate.py: cfg 4,
# 28 steps, fp32 stream, 3 aspects × 16 captions).
DEFAULT_GAMMA_PATH = os.path.join(_HERE, "cns_gamma.npz")

CNS_SAMPLER_NAME = "er_sde_cns"


# ---------------------------------------------------------------------------
# Recolorer — numerics imported from the shared anima_lora core (resolved against
# the live tree or the ``_vendor/`` subset by the sys.path bootstrap in
# __init__.py). Only the γ-artifact path is node-specific, so this subclass adds
# just ``from_path``.
# ---------------------------------------------------------------------------

from library.inference.corrections.cns_core import (
    CNSRecolorer as _CNSRecolorerCore,
)
from library.inference.corrections.cns_core import radial_bins  # noqa: F401


class CNSRecolorer(_CNSRecolorerCore):
    """CNS recolorer with the node's bundled γ as ``"auto"``."""

    @classmethod
    def from_path(cls, path: str, strength: float = 1.0) -> "CNSRecolorer":
        """Load from an npz path, or the literal ``"auto"`` (the bundled γ)."""
        resolved = DEFAULT_GAMMA_PATH if path == "auto" else path
        return cls.from_npz(resolved, strength=strength)


# ---------------------------------------------------------------------------
# MOSS-Image: its own γ, bundled (calibrated in anima_lora,
# bench/moss_spectrum/cns_calibrate.py, shift 5, 50 steps), one per variant and
# cfg, picked by the variant fingerprint and the guider's nearest cfg, and the
# FFT on the un-patchified 32 ch /8 grid (anima_lora library/moss/cns.py). On the
# packed 128 ch grid the top octave of the FLUX.2 latent sits in the 2×2 phase
# channels, out of reach of a spatial filter. ComfyUI's Flux2 VAE packs
# (c pi pj), the same order.
# ---------------------------------------------------------------------------

# (variant, cfg) → γ; mirrors anima_lora library.moss.cns.GAMMA_PATHS.
MOSS_GAMMAS = {
    ("base", 3.0): "moss_cns_gamma.npz",
    ("base", 4.0): "moss_cns_gamma_cfg4.npz",
    ("base", 5.0): "moss_cns_gamma_cfg5.npz",
    ("sft", 3.0): "moss_cns_gamma_sft_cfg3.npz",
    ("sft", 4.0): "moss_cns_gamma_sft_cfg4.npz",
    ("sft", 5.0): "moss_cns_gamma_sft_cfg5.npz",
}
MOSS_GAMMA_PATHS = {
    f"moss_{v}_cfg{c:g}": os.path.join(_HERE, f) for (v, c), f in MOSS_GAMMAS.items()
}
# Base and SFT differ in the transformer only; time_embedding.2.bias moves ~2 %
# between them, well clear of bf16/fp8 rounding. Score = projection of the live
# bias onto (SFT − Base): 0 for Base, 1 for SFT.
_FINGERPRINT_PATH = os.path.join(_HERE, "moss_variant_fingerprint.npz")
_FINGERPRINT: dict = {}


def _unpatchify(z: torch.Tensor) -> torch.Tensor:
    """``(B,128,h,w)`` → ``(B,32,2h,2w)``."""
    b, c, h, w = z.shape
    return (
        z.reshape(b, c // 4, 2, 2, h, w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(b, c // 4, 2 * h, 2 * w)
    )


def _patchify(z: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_unpatchify`."""
    b, c, h, w = z.shape
    return (
        z.view(b, c, h // 2, 2, w // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(b, c * 4, h // 2, w // 2)
    )


class _MossRecolorer:
    def __init__(self, core: CNSRecolorer):
        self.core = core

    def recolor(self, white: torch.Tensor, sigma_s: float) -> torch.Tensor:
        return _patchify(self.core.recolor(_unpatchify(white), sigma_s))


def _is_moss(model) -> bool:
    try:
        dit = model.inner_model.model_patcher.get_model_object("diffusion_model")
    except Exception:  # noqa: BLE001 - unknown wrapper: treat as Anima
        return False
    return hasattr(dit, "video_head_modulation") and hasattr(dit, "final_norm_video")


def _moss_variant(dit) -> str:
    """``base`` or ``sft`` from the live weights; ``base`` if they can't be read."""
    if not _FINGERPRINT:
        d = np.load(_FINGERPRINT_PATH)
        _FINGERPRINT.update(base=torch.from_numpy(d["base"]), sft=torch.from_numpy(d["sft"]))
    base, sft = _FINGERPRINT["base"], _FINGERPRINT["sft"]
    try:
        x = dit.time_embedding[2].bias.detach().float().cpu()
    except Exception as e:  # noqa: BLE001 - meta / offloaded weight
        logger.warning("CNS: can't read MOSS weights for the variant check (%s); using Base γ", e)
        return "base"
    if x.shape != base.shape:
        return "base"
    d = sft - base
    score = float(((x - base) * d).sum() / (d * d).sum())
    return "sft" if score >= 0.5 else "base"


def _gamma_family(model) -> str:
    """``anima``, or the MOSS γ for the model's variant at the calibrated cfg
    nearest the guider's (``model.inner_model`` is the ``CFGGuider``; a guider
    without ``cfg`` gets the cfg 3 γ)."""
    if not _is_moss(model):
        return "anima"
    dit = model.inner_model.model_patcher.get_model_object("diffusion_model")
    variant = _moss_variant(dit)
    cfg = getattr(model.inner_model, "cfg", None)
    if not isinstance(cfg, (int, float)):
        cfg = 3.0
    nearest = min((c for v, c in MOSS_GAMMAS if v == variant), key=lambda c: abs(c - cfg))
    return f"moss_{variant}_cfg{nearest:g}"


# ---------------------------------------------------------------------------
# Lazy γ per model family — load each npz once, build a fresh recolorer per
# sample (so a per-sample aspect change can't get pinned by a stale aspect lock).
# ---------------------------------------------------------------------------

_GAMMA_ARRAYS: dict = {}
_GAMMA_LOCK = threading.Lock()


def _load_gamma_arrays(family: str) -> dict:
    if family not in _GAMMA_ARRAYS:
        with _GAMMA_LOCK:
            if family not in _GAMMA_ARRAYS:
                path = MOSS_GAMMA_PATHS.get(family) or DEFAULT_GAMMA_PATH
                d = np.load(path)
                _GAMMA_ARRAYS[family] = {k: d[k] for k in ("gamma", "aspects", "sigmas")}
                logger.info("CNS: %s γ from %s", family, path)
    return _GAMMA_ARRAYS[family]


def _make_recolorer(strength: float = 1.0, family: str = "anima"):
    a = _load_gamma_arrays(family)
    core = CNSRecolorer(a["gamma"], a["aspects"], a["sigmas"], strength=strength)
    return _MossRecolorer(core) if family in MOSS_GAMMA_PATHS else core


# ---------------------------------------------------------------------------
# Sampler — copy of comfy.k_diffusion.sampling.sample_er_sde with the single
# noise-injection line routed through the recolorer.
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_er_sde_cns(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    noise_sampler=None,
    noise_scaler=None,
    max_stage=3,
):
    """ER-SDE-Solver-3 (arXiv 2309.06169) with CNS-recolored per-step noise.

    Verbatim copy of ``comfy.k_diffusion.sampling.sample_er_sde``; the only
    change is the final stochastic-injection line, where the white draw is
    recolored via :class:`CNSRecolorer` at σ = ``float(sigmas[i])`` (matching the
    CLI ``ERSDESampler._sample_noise`` seam). With recoloring removed the math is
    bit-identical to stock ``er_sde``.
    """
    from comfy.k_diffusion.sampling import (
        default_noise_sampler,
        offset_first_sigma_for_snr,
        sigma_to_half_log_snr,
    )
    from comfy.utils import model_trange as trange

    recolorer = _make_recolorer(strength=1.0, family=_gamma_family(model))

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    def default_er_sde_noise_scaler(x):
        return x * ((x ** 0.3).exp() + 10.0)

    noise_scaler = default_er_sde_noise_scaler if noise_scaler is None else noise_scaler
    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    half_log_snrs = sigma_to_half_log_snr(sigmas, model_sampling)
    er_lambdas = half_log_snrs.neg().exp()  # er_lambda_t = sigma_t / alpha_t

    old_denoised = None
    old_denoised_d = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1 Euler
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                # Stage 2
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3:
                    # Stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                # CNS seam: recolor the white draw before injection (σ = sigmas[i],
                # matching anima_lora ERSDESampler._sample_noise). Variance-conserving.
                white = noise_sampler(sigmas[i], sigmas[i + 1])
                colored = recolorer.recolor(white, float(sigmas[i]))
                x = x + alpha_t * colored * s_noise * (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)

        old_denoised = denoised
    return x


# ---------------------------------------------------------------------------
# Registration — make `er_sde_cns` selectable in the sampler_name dropdown
# ---------------------------------------------------------------------------


def register_cns_sampler() -> None:
    """Register ``er_sde_cns`` into ComfyUI's global sampler registry.

    ``comfy.samplers.ksampler(name)`` dispatches via
    ``getattr(comfy.k_diffusion.sampling, "sample_" + name)``, and the dropdown
    reads the live ``KSampler.SAMPLERS`` list — so we set the attribute and
    append the name. Idempotent; called once at node import.
    """
    try:
        import comfy.samplers
        import comfy.k_diffusion.sampling as kds

        if not hasattr(kds, "sample_er_sde_cns"):
            kds.sample_er_sde_cns = sample_er_sde_cns
        samplers = comfy.samplers.KSampler.SAMPLERS
        if CNS_SAMPLER_NAME not in samplers:
            samplers.append(CNS_SAMPLER_NAME)
        logger.info("CNS: registered sampler '%s' (colored ER-SDE; Anima / MOSS γ)", CNS_SAMPLER_NAME)
    except Exception as e:  # noqa: BLE001 — never break the node pack over this
        logger.warning("CNS: failed to register '%s' sampler (%s)", CNS_SAMPLER_NAME, e)
