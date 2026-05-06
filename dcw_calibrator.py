"""DCW online calibrator — per-step λ via a fusion-head MLP.

Vendored from ``anima_lora/library/inference/dcw_calibrator.py`` plus the
shared ``FusionHead`` / ``haar_LL_norm`` from ``anima_lora/networks/dcw.py``,
so this node has no runtime dep on the trainer repo.

The calibrator is loaded from a safetensors artifact (head weights +
standardization stats), observes the LL-band Haar norm of the post-CFG
velocity over the first ``k_warmup`` steps, fires the MLP at step
``k_warmup`` to predict per-prompt λ̂*_p, then applies::

    λ_i = baseline_lambda · (1 − σ_i)                                      [all i]
        + α̂ · gain · (1 − σ_i)         for target_start ≤ i < target_end

clamped to ±0.05.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

import folder_paths

from .mod_guidance import _extract_raw_and_t5

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATOR_FILENAME = "fusion_head-0506.safetensors"
DEFAULT_CALIBRATOR_URL = (
    "https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler/releases/download/"
    "0429/fusion_head-0506.safetensors"
)
DEFAULT_CALIBRATOR_SUBDIR = "anima_dcw_calibrator"

_VALID_SCHEMAS = ("dcw_v5_lambda_scalar", "dcw_v4_fusion_head")
_LAMBDA_CLAMP = 0.05

_DOWNLOAD_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Auto-download
# ---------------------------------------------------------------------------


def get_default_calibrator_path() -> str:
    """Return local path to the default fusion-head artifact, downloading if missing."""
    target_dir = os.path.join(folder_paths.models_dir, DEFAULT_CALIBRATOR_SUBDIR)
    target_path = os.path.join(target_dir, DEFAULT_CALIBRATOR_FILENAME)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return target_path

    with _DOWNLOAD_LOCK:
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return target_path
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"DCW calibrator: cannot create directory {target_dir} ({e}). "
                f"If ComfyUI is installed under Program Files, move it or run as admin. "
                f"Otherwise download manually from {DEFAULT_CALIBRATOR_URL} and place it at {target_path}."
            ) from e
        tmp_path = target_path + ".download"
        logger.info(
            f"DCW calibrator: downloading default fusion head from {DEFAULT_CALIBRATOR_URL}"
        )
        try:
            req = urllib.request.Request(
                DEFAULT_CALIBRATOR_URL,
                headers={"User-Agent": "comfyui-spectrum/dcw_calibrator"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                downloaded = 0
                next_log = 2 * 1024 * 1024
                with open(tmp_path, "wb") as fh:
                    while True:
                        chunk = resp.read(128 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_log:
                            if total:
                                logger.info(
                                    f"DCW calibrator: {downloaded // (1024 * 1024)}MB "
                                    f"/ {total // (1024 * 1024)}MB"
                                )
                            else:
                                logger.info(
                                    f"DCW calibrator: {downloaded // (1024 * 1024)}MB"
                                )
                            next_log += 2 * 1024 * 1024
                if total and downloaded != total:
                    raise RuntimeError(
                        f"truncated download: got {downloaded} of {total} bytes"
                    )
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(
                f"DCW calibrator: failed to download from {DEFAULT_CALIBRATOR_URL} ({e}). "
                f"If this is a corporate network or TLS-intercepting proxy, try `pip install -U certifi`. "
                f"Otherwise download manually and place the file at {target_path}."
            ) from e
        last_err: Optional[Exception] = None
        for attempt in range(5):
            try:
                os.replace(tmp_path, target_path)
                last_err = None
                break
            except PermissionError as e:
                last_err = e
                time.sleep(0.2 * (attempt + 1))
        if last_err is not None:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(
                f"DCW calibrator: downloaded but could not rename into place ({last_err}). "
                f"This is usually Windows antivirus holding the file open. "
                f"Try adding {target_dir} to your AV exclusions, or download manually from "
                f"{DEFAULT_CALIBRATOR_URL} and place it at {target_path}."
            ) from last_err
        logger.info(f"DCW calibrator: saved to {target_path}")
        return target_path


# ---------------------------------------------------------------------------
# Haar LL norm (matches anima_lora/networks/dcw.py)
# ---------------------------------------------------------------------------


def _haar_dwt_2d(
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def haar_LL_norm(v: torch.Tensor) -> float:
    """Single-level Haar LL-band Frobenius norm of a velocity tensor."""
    LL, _, _, _ = _haar_dwt_2d(v.float())
    return float(LL.flatten().norm())


# ---------------------------------------------------------------------------
# FusionHead (matches anima_lora/networks/dcw.py)
# ---------------------------------------------------------------------------


class FusionHead(nn.Module):
    """v4/v5 fusion head: (c_pool, g_obs[0:k], aux) → (α̂, log σ̂²)."""

    def __init__(
        self,
        c_pool_dim: int = 1024,
        k: int = 7,
        aux_dim: int = 3,
        c_proj_dim: int = 0,
        hidden: Tuple[int, ...] = (256, 128),
        sigma_hidden: int = 64,
        dropout: float = 0.0,
        log_sigma2_init: float = 0.0,
    ):
        super().__init__()
        self.k = k
        self.c_pool_dim = c_pool_dim
        self.c_proj_dim = c_proj_dim
        if c_proj_dim > 0:
            self.c_proj = nn.Sequential(
                nn.LayerNorm(c_pool_dim),
                nn.Linear(c_pool_dim, c_proj_dim),
            )
            cat_dim = c_proj_dim
        else:
            self.c_proj = nn.Identity()
            cat_dim = c_pool_dim
        in_dim = cat_dim + k + aux_dim

        alpha_layers: List[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for h in hidden:
            alpha_layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        alpha_out = nn.Linear(prev, 1)
        nn.init.zeros_(alpha_out.weight)
        nn.init.zeros_(alpha_out.bias)
        alpha_layers.append(alpha_out)
        self.alpha_mlp = nn.Sequential(*alpha_layers)

        sigma_out = nn.Linear(sigma_hidden, 1)
        nn.init.zeros_(sigma_out.weight)
        nn.init.zeros_(sigma_out.bias)
        with torch.no_grad():
            sigma_out.bias.fill_(log_sigma2_init)
        self.sigma_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, sigma_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            sigma_out,
        )

    def forward(self, c_pool, g_obs, aux):
        c = self.c_proj(c_pool)
        x = torch.cat([c, g_obs, aux], dim=-1)
        return self.alpha_mlp(x).squeeze(-1), self.sigma_mlp(x).squeeze(-1)


# ---------------------------------------------------------------------------
# OnlineDCWCalibrator (matches anima_lora/library/inference/dcw_calibrator.py)
# ---------------------------------------------------------------------------


class OnlineDCWCalibrator:
    def __init__(
        self,
        head: FusionHead,
        centroid: torch.Tensor,
        aux_mean: torch.Tensor,
        aux_std: torch.Tensor,
        g_obs_mean: torch.Tensor,
        g_obs_std: torch.Tensor,
        k_warmup: int,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        target_start: Optional[int] = None,
        target_end: Optional[int] = None,
        c_pool_norm: str = "none",
        c_pool_mean: Optional[torch.Tensor] = None,
        c_pool_std: Optional[torch.Tensor] = None,
        baseline_lambda: float = 0.0,
    ):
        self.head = head.to(device=device, dtype=dtype).eval()
        self.centroid = centroid.to(device=device, dtype=dtype)
        self.aux_mean = aux_mean.to(device=device, dtype=dtype)
        self.aux_std = aux_std.to(device=device, dtype=dtype)
        self.g_obs_mean = g_obs_mean.to(device=device, dtype=dtype)
        self.g_obs_std = g_obs_std.to(device=device, dtype=dtype)
        self.k_warmup = int(k_warmup)
        self.n_steps = int(n_steps)
        self.target_start = int(k_warmup if target_start is None else target_start)
        self.target_end = int(n_steps if target_end is None else target_end)
        self.device = device
        self.dtype = dtype
        self.c_pool_norm = c_pool_norm
        self.c_pool_mean = (
            c_pool_mean.to(device=device, dtype=dtype)
            if c_pool_mean is not None
            else None
        )
        self.c_pool_std = (
            c_pool_std.to(device=device, dtype=dtype)
            if c_pool_std is not None
            else None
        )
        self.is_active: bool = False
        self.c_pool: Optional[torch.Tensor] = None
        self.aux: Optional[torch.Tensor] = None
        self.g_obs_buf: List[float] = []
        self.alpha_eff: float = 0.0
        self.gain: float = 1.0
        self.baseline_lambda: float = float(baseline_lambda)

    @classmethod
    def from_safetensors(
        cls, path, *, device: torch.device
    ) -> "OnlineDCWCalibrator":
        path = Path(path)
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        schema = meta.get("schema")
        if schema not in _VALID_SCHEMAS:
            raise ValueError(
                f"{path}: unexpected schema {schema!r}, expected one of {_VALID_SCHEMAS}"
            )
        target_kind = meta.get("target_kind", "lambda_scalar")
        if target_kind != "lambda_scalar":
            raise ValueError(
                f"{path}: target_kind={target_kind!r} is not supported by this node — "
                "use a current `make dcw-train` artifact (always lambda_scalar)."
            )
        k_warmup = int(meta.get("k_warmup", 7))
        n_steps = int(meta.get("n_steps", 28))
        target_start = int(meta.get("target_start", k_warmup))
        target_end = int(meta.get("target_end", n_steps))
        baseline_lambda = float(meta.get("baseline_lambda", 0.0))

        head_sd = {
            k[len("head.") :]: v for k, v in tensors.items() if k.startswith("head.")
        }
        if "alpha_mlp.0.weight" not in head_sd:
            raise ValueError(
                f"{path}: missing 'head.alpha_mlp.*' keys — predates the alpha/sigma "
                "trunk split. Retrain with `make dcw-train`."
            )
        in_dim = int(head_sd["alpha_mlp.0.weight"].shape[0])
        if "aspect_emb.weight" in head_sd:
            raise ValueError(
                f"{path}: artifact contains 'aspect_emb.weight' — predates the "
                "bucket-cosmetic removal. Retrain with `make dcw-train`."
            )
        aux_dim = 3
        if "c_proj.1.weight" in head_sd:
            c_proj_w = head_sd["c_proj.1.weight"]
            c_proj_dim = int(c_proj_w.shape[0])
            c_pool_dim = int(c_proj_w.shape[1])
            cat_dim = c_proj_dim
        else:
            c_proj_dim = 0
            c_pool_dim = in_dim - (k_warmup + aux_dim)
            cat_dim = c_pool_dim
        if cat_dim + k_warmup + aux_dim != in_dim:
            raise ValueError(
                f"{path}: shape mismatch — cat({cat_dim}) + k({k_warmup}) "
                f"+ aux({aux_dim}) != alpha_mlp.0 in_dim({in_dim}). "
                "Likely a pre-cleanup artifact; retrain."
            )
        head = FusionHead(
            c_pool_dim=c_pool_dim,
            k=k_warmup,
            aux_dim=aux_dim,
            c_proj_dim=c_proj_dim,
        )
        head.load_state_dict(head_sd, strict=False)

        c_pool_norm = meta.get("c_pool_norm", "none")
        if c_pool_norm not in ("none", "l2", "standardize", "l2_then_standardize"):
            raise ValueError(
                f"{path}: unknown c_pool_norm={c_pool_norm!r}. Update the node."
            )
        ctrl = cls(
            head=head,
            centroid=tensors["centroid_c_pool"],
            aux_mean=tensors["aux_mean"],
            aux_std=tensors["aux_std"],
            g_obs_mean=tensors["g_obs_mean"],
            g_obs_std=tensors["g_obs_std"],
            k_warmup=k_warmup,
            n_steps=n_steps,
            device=device,
            target_start=target_start,
            target_end=target_end,
            c_pool_norm=c_pool_norm,
            c_pool_mean=tensors.get("c_pool_mean"),
            c_pool_std=tensors.get("c_pool_std"),
            baseline_lambda=baseline_lambda,
        )
        logger.info(
            "DCW calibrator: loaded %s (schema=%s, k=%d, target=[%d:%d], "
            "%d steps, c_pool_norm=%s, baseline_lambda=%+.4g)",
            path.name,
            schema,
            k_warmup,
            target_start,
            target_end,
            n_steps,
            c_pool_norm,
            baseline_lambda,
        )
        return ctrl

    def setup(self, embed: torch.Tensor, embed_mask: Optional[torch.Tensor], *, gain: float = 1.0) -> None:
        """Compute c_pool + aux for this generation. Idempotent."""
        self.is_active = False
        self.g_obs_buf = []
        self.alpha_eff = 0.0
        self.gain = float(gain)

        e = embed[0].to(self.device, dtype=self.dtype)
        if embed_mask is not None:
            mask = embed_mask[0].to(self.device, dtype=torch.bool)
            valid = e[mask]
            cap_len = int(mask.sum().item())
        else:
            valid = e
            cap_len = e.shape[0]
        if valid.numel() == 0:
            logger.warning("DCW calibrator: empty embed mask — disabling")
            return

        c_pool_raw = valid.mean(dim=0)
        token_l2 = valid.norm(dim=-1)
        cos_centroid = float(
            torch.dot(c_pool_raw, self.centroid)
            / (c_pool_raw.norm() * self.centroid.norm() + 1e-9)
        )
        aux_raw = torch.tensor(
            [float(cap_len), cos_centroid, float(token_l2.std().item())],
            device=self.device,
            dtype=self.dtype,
        )
        c_pool = c_pool_raw
        if self.c_pool_norm in ("l2", "l2_then_standardize"):
            c_pool = c_pool / (c_pool.norm() + 1e-9)
        if self.c_pool_norm in ("standardize", "l2_then_standardize"):
            if self.c_pool_mean is None or self.c_pool_std is None:
                raise RuntimeError(
                    "c_pool_norm requests standardize but artifact has no "
                    "c_pool_mean / c_pool_std tensors — retrain to ship them."
                )
            c_pool = (c_pool - self.c_pool_mean) / self.c_pool_std
        self.c_pool = c_pool
        self.aux = (aux_raw - self.aux_mean) / self.aux_std
        self.is_active = True
        logger.info(
            "DCW calibrator: setup target=[%d:%d] gain=%.4g baseline=%+.4g "
            "cap_len=%d cos_centroid=%.3f c_pool_norm=%s",
            self.target_start,
            self.target_end,
            self.gain,
            self.baseline_lambda,
            cap_len,
            cos_centroid,
            self.c_pool_norm,
        )

    def record(self, step_i: int, noise_pred: torch.Tensor) -> None:
        """Observe LL-band norm of the post-CFG velocity at warmup steps."""
        if not self.is_active or step_i >= self.k_warmup:
            return
        self.g_obs_buf.append(haar_LL_norm(noise_pred))

    def fire_head_if_due(self, step_i: int) -> None:
        if not self.is_active or step_i != self.k_warmup:
            return
        if len(self.g_obs_buf) < self.k_warmup:
            logger.warning(
                "DCW calibrator: only %d/%d warmup obs collected — disabling",
                len(self.g_obs_buf),
                self.k_warmup,
            )
            self.alpha_eff = 0.0
            return
        g_obs = torch.tensor(
            self.g_obs_buf[: self.k_warmup], device=self.device, dtype=self.dtype
        )
        g_obs_n = (g_obs - self.g_obs_mean) / self.g_obs_std
        with torch.no_grad():
            alpha_hat, _ = self.head(
                self.c_pool.unsqueeze(0),
                g_obs_n.unsqueeze(0),
                self.aux.unsqueeze(0),
            )
        self.alpha_eff = float(alpha_hat[0].item())
        logger.info(
            "DCW calibrator: head fired at step %d — α̂=%+.4g",
            step_i,
            self.alpha_eff,
        )

    def lambda_for_step(self, step_i: int, sigma_i: float) -> float:
        if not self.is_active:
            return 0.0
        env = 1.0 - sigma_i
        lam_i = self.baseline_lambda * env
        if self.target_start <= step_i < self.target_end:
            lam_i += self.alpha_eff * self.gain * env
        return max(-_LAMBDA_CLAMP, min(_LAMBDA_CLAMP, lam_i))


# ---------------------------------------------------------------------------
# Setup helper called from the node's sample()
# ---------------------------------------------------------------------------


def _resolve_calibrator_path(name: Optional[str]) -> str:
    from .nodes import AUTO_CALIBRATOR_SENTINEL  # avoid cycle at import time

    if name in (None, "", AUTO_CALIBRATOR_SENTINEL):
        return get_default_calibrator_path()
    path = folder_paths.get_full_path("loras", name)
    if path is None:
        raise RuntimeError(f"DCW calibrator artifact not found: {name}")
    return path


def setup_dcw_calibrator(
    model_clone,
    clip,
    positive,
    calibrator_name: Optional[str],
    *,
    gain: float = 1.0,
) -> Optional[OnlineDCWCalibrator]:
    """Load the calibrator + run setup() with c_pool from the post-LLM-adapter
    pooling of the positive prompt. Returns the calibrator (active) or ``None``
    if the artifact failed to load or setup hit an empty embed.

    Mirrors ``anima_lora/library/inference/generation.py``'s setup path: feed the
    raw positive cond + t5xxl meta through ``dit.preprocess_text_embeds`` to
    recover the same (B, L, 1024) tensor the trainer cached as
    ``crossattn_emb_v0``. ``embed_mask`` is reconstructed from per-token L2 norm
    > 0 (the LLM adapter zero-pads to 512 and ``t5xxl_weights`` zero out
    dropped tokens), which matches the trainer's ``attn_mask_v0`` to within the
    cap_len aux-feature tolerance.
    """
    try:
        path = _resolve_calibrator_path(calibrator_name)
    except Exception as e:
        logger.warning("DCW calibrator: cannot resolve artifact: %s — disabling", e)
        return None

    dm = model_clone.model.diffusion_model
    device = next(dm.parameters()).device
    dtype = model_clone.model.get_dtype_inference()

    try:
        calibrator = OnlineDCWCalibrator.from_safetensors(path, device=device)
    except Exception as e:
        logger.warning("DCW calibrator: failed to load %s: %s — disabling", path, e)
        return None

    pos_raw, pos_t5_ids, pos_t5_weights = _extract_raw_and_t5(positive)
    raw = pos_raw.unsqueeze(0).to(device=device, dtype=dtype)
    t5_ids = pos_t5_ids.unsqueeze(0).to(device=device) if pos_t5_ids is not None else None
    t5_weights = (
        pos_t5_weights.unsqueeze(0).unsqueeze(-1).to(device=device, dtype=dtype)
        if pos_t5_weights is not None
        else None
    )
    with torch.no_grad():
        adapted = dm.preprocess_text_embeds(raw, t5_ids, t5xxl_weights=t5_weights)
    # adapted: (1, 512, 1024) post-LLM-adapter, zero-padded.
    embed_mask = (adapted.float().norm(dim=-1) > 1e-6)  # (1, 512) bool
    calibrator.setup(embed=adapted.float(), embed_mask=embed_mask, gain=gain)
    if not calibrator.is_active:
        return None
    return calibrator
