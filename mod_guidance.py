"""Modulation guidance: adapter loading, projection, APPLY_MODEL wrapper.

The wrapper runs *outside* the torch.compile region (at the APPLY_MODEL layer,
before `_apply_model` dispatches to the compiled `diffusion_model`). All tensor
math — projection, guidance delta, per-batch assembly — is precomputed once per
sample and only a cached tensor is written to `t_embedding_norm._anima_mod_delta`
on the hot path. Inside compile, the forward hook does nothing more than
`output + delta`, which dynamo inlines cleanly.
"""

import logging
import os
import threading
import time
import urllib.request
import weakref
from typing import List, Optional

import torch

import comfy.patcher_extension
import folder_paths

from library.inference.corrections.mod_guidance_core import (
    build_block_schedule,
    project_pooled,
)

logger = logging.getLogger(__name__)

# Printed once at import (ComfyUI server startup). If you don't see this line in
# the startup log, the server is running a stale copy — restart it.
logger.info("[mod-guidance] module loaded: sigma-film v5 (fp32 head, two-step t_emb)")

PROJ_KEYS = ("0.weight", "0.bias", "2.weight", "2.bias")
# Optional σ-FiLM generator (timestep-conditioned mod head). Present only in
# adapters trained with `--mod_sigma_film`; absent ⇒ plain σ-flat head.
FILM_KEYS = ("sigma_film.weight", "sigma_film.bias")
MOD_APPLY_WRAPPER_KEY = "spectrum_mod_guidance_apply"
MOD_DIFFUSION_WRAPPER_KEY = "spectrum_mod_guidance"  # legacy key for cleanup
MOD_STATE_KEY = "spectrum_mod_guidance_state"

AUTO_ADAPTER_SENTINEL = "(auto-download default)"
DEFAULT_ADAPTER_FILENAME = "pooled_text_proj-0611.safetensors"
DEFAULT_ADAPTER_URL = (
    "https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler/releases/download/"
    "0605/pooled_text_proj-0611.safetensors"
)
DEFAULT_ADAPTER_SUBDIR = "anima_mod_guidance"

_ADAPTER_LOCK = threading.Lock()
_ADAPTER_CPU_CACHE: dict = {}
_ADAPTER_TYPED_CACHE: dict = {}
_DOWNLOAD_LOCK = threading.Lock()


def get_default_adapter_path() -> str:
    """Return local path to the default pooled_text_proj adapter, downloading if missing."""
    target_dir = os.path.join(folder_paths.models_dir, DEFAULT_ADAPTER_SUBDIR)
    target_path = os.path.join(target_dir, DEFAULT_ADAPTER_FILENAME)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return target_path

    with _DOWNLOAD_LOCK:
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return target_path
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"Mod guidance: cannot create adapter directory {target_dir} ({e}). "
                f"If ComfyUI is installed under Program Files, move it or run as admin. "
                f"Otherwise download manually from {DEFAULT_ADAPTER_URL} and place it at {target_path}."
            ) from e
        tmp_path = target_path + ".download"
        logger.info(
            f"Mod guidance: downloading default adapter (~12MB) from {DEFAULT_ADAPTER_URL}"
        )
        try:
            req = urllib.request.Request(
                DEFAULT_ADAPTER_URL,
                headers={"User-Agent": "comfyui-spectrum/mod_guidance"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                downloaded = 0
                next_log = 2 * 1024 * 1024  # log every 2MB
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
                                    f"Mod guidance: {downloaded // (1024 * 1024)}MB "
                                    f"/ {total // (1024 * 1024)}MB"
                                )
                            else:
                                logger.info(
                                    f"Mod guidance: {downloaded // (1024 * 1024)}MB"
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
                f"Mod guidance: failed to download adapter from {DEFAULT_ADAPTER_URL} ({e}). "
                f"If this is a corporate network or TLS-intercepting proxy, try `pip install -U certifi`. "
                f"Otherwise download manually and place the file at {target_path}."
            ) from e
        # Windows: antivirus may briefly lock tmp_path on close, causing os.replace
        # to raise PermissionError ([WinError 5]). Retry a few times with backoff.
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
                f"Mod guidance: downloaded adapter but could not rename into place ({last_err}). "
                f"This is usually Windows antivirus holding the file open. "
                f"Try adding {target_dir} to your AV exclusions, or download manually from "
                f"{DEFAULT_ADAPTER_URL} and place it at {target_path}."
            ) from last_err
        logger.info(f"Mod guidance: saved adapter to {target_path}")
        return target_path


def _resolve_adapter_path(adapter_name: Optional[str]) -> str:
    if adapter_name in (None, "", AUTO_ADAPTER_SENTINEL):
        return get_default_adapter_path()
    path = folder_paths.get_full_path("loras", adapter_name)
    if path is None:
        raise RuntimeError(f"Adapter not found: {adapter_name}")
    return path


def _load_adapter_cpu(path: str) -> dict:
    path = os.path.abspath(path)
    with _ADAPTER_LOCK:
        if path in _ADAPTER_CPU_CACHE:
            return _ADAPTER_CPU_CACHE[path]

    from safetensors.torch import load_file

    raw = load_file(path)
    missing = [k for k in PROJ_KEYS if k not in raw]
    if missing:
        raise RuntimeError(
            f"Adapter missing keys: {', '.join(missing)}. "
            f"Expected pooled_text_proj format with keys: {PROJ_KEYS}"
        )
    keys = list(PROJ_KEYS)
    if all(k in raw for k in FILM_KEYS):
        keys += list(FILM_KEYS)  # σ-FiLM adapter — carry the generator too
    state = {k: raw[k].detach().float().cpu().contiguous() for k in keys}
    with _ADAPTER_LOCK:
        _ADAPTER_CPU_CACHE[path] = state
    return state


def _adapter_has_film(state) -> bool:
    return all(k in state for k in FILM_KEYS)


def _get_adapter_typed(path: str, device, dtype):
    path = os.path.abspath(path)
    key = (path, str(device), str(dtype))
    with _ADAPTER_LOCK:
        if key in _ADAPTER_TYPED_CACHE:
            return _ADAPTER_TYPED_CACHE[key]
    cpu = _load_adapter_cpu(path)
    typed = {k: v.to(device=device, dtype=dtype) for k, v in cpu.items()}
    with _ADAPTER_LOCK:
        _ADAPTER_TYPED_CACHE[key] = typed
    return typed


def _project(pooled, adapter_state):
    """σ-flat projection through the 2-layer MLP (Linear -> SiLU -> Linear).

    Thin seam over the shared ``project_pooled`` core — the node loads the head as
    a raw safetensors state dict (PROJ_KEYS), so it unpacks the weights here.
    """
    return project_pooled(
        pooled,
        adapter_state["0.weight"],
        adapter_state["0.bias"],
        adapter_state["2.weight"],
        adapter_state["2.bias"],
    )


def _project_film(pooled, adapter_state, t_emb, out_dtype=None):
    """σ-conditioned projection. With σ-FiLM weights (sigma_film.*) and a (B, C)
    normed time embedding ``t_emb``, FiLM-modulate the hidden between the two
    linears; without either, falls back to the σ-flat head. Delegates the math to
    the shared ``project_pooled`` core (single source of truth with
    ``library/anima/models.py::_pooled_text_delta``); this seam only unpacks the
    raw state dict and routes ``out_dtype`` (run the fp32 head, cast back to the
    block compute dtype)."""
    has_film = "sigma_film.weight" in adapter_state
    return project_pooled(
        pooled,
        adapter_state["0.weight"],
        adapter_state["0.bias"],
        adapter_state["2.weight"],
        adapter_state["2.bias"],
        film_w=adapter_state["sigma_film.weight"] if has_film else None,
        film_b=adapter_state.get("sigma_film.bias") if has_film else None,
        t_emb=t_emb,
        out_dtype=out_dtype,
    )


def _compute_t_emb(dit, t):
    """Reproduce the DiT's normed time embedding (the FiLM conditioning) for the
    current step's timestep. Returns (B, C). Raises on any shape / attribute
    mismatch so the caller can fall back to the σ-flat path.

    Two subtleties of MiniTrainDIT, both required:
    1. Its ``Timesteps`` (sinusoid) returns fp32 and never casts back, so we must
       step the two t_embedder stages by hand and insert the cast ourselves —
       calling the whole Sequential clashes fp32 sinusoid × bf16 linear.
    2. The cast target is the t_embedder's WEIGHT dtype, NOT the incoming latent
       dtype: in the APPLY_MODEL wrapper ``args[0]`` is still the sampler's fp32
       latent (the model casts it internally later), so x.dtype is wrong here."""
    te = dit.t_embedder
    p = next(te[1].parameters(), None)
    wdt = p.dtype if p is not None else t.dtype
    t_bt = t.reshape(-1, 1) if t.ndim == 1 else t
    sin = te[0](t_bt)  # Timesteps -> fp32
    emb = te[1](sin.to(wdt))  # cast to embedder weight dtype, then its linears
    if isinstance(emb, (tuple, list)):
        emb = emb[0]
    emb = dit.t_embedding_norm(emb)
    if emb.ndim == 3:
        emb = emb[:, 0, :]
    return emb


class _PerBlockState:
    """Runtime state read by per-block / final-layer pre-hooks.

    Lives on `dit._anima_mod_block_state` for the duration of a single
    `_mod_apply_wrapper` call. Hooks check for `None` to fast-no-op.
    """

    __slots__ = ("delta_unit", "schedule", "final_w")

    def __init__(self, delta_unit: torch.Tensor, schedule: List[float], final_w: float):
        self.delta_unit = delta_unit  # (1, C) — proj_tag - proj_neg
        self.schedule = schedule  # length num_blocks; w_l per block
        self.final_w = final_w  # scalar w applied to final_layer


class ModGuidanceState:
    """Per-sample state. Holds raw text tensors at construction time and caches
    the precomputed base projections + steering delta after the first forward.

    All hot-path work happens in the APPLY_MODEL wrapper, outside torch.compile.
    The compiled DiT forward only sees a single tensor read inside the t_emb hook
    (base projection) plus per-block pre-hooks that add `w(ℓ) * delta_unit`.
    """

    def __init__(
        self,
        adapter_path: str,
        w: float,
        tag_raw: torch.Tensor,
        tag_t5_ids: Optional[torch.Tensor],
        tag_t5_weights: Optional[torch.Tensor],
        neg_raw: torch.Tensor,
        neg_t5_ids: Optional[torch.Tensor],
        neg_t5_weights: Optional[torch.Tensor],
        pos_raw: torch.Tensor,
        pos_t5_ids: Optional[torch.Tensor],
        pos_t5_weights: Optional[torch.Tensor],
        qneg_raw: torch.Tensor,
        qneg_t5_ids: Optional[torch.Tensor],
        qneg_t5_weights: Optional[torch.Tensor],
        dit,
        start_layer: int = 0,
        end_layer: int = 27,
        taper: int = 0,
        taper_scale: float = 0.25,
        final_w: float = 0.0,
    ):
        self.adapter_path = adapter_path
        self.w = w
        self.tag_raw = tag_raw
        self.tag_t5_ids = tag_t5_ids
        self.tag_t5_weights = tag_t5_weights
        self.neg_raw = neg_raw
        self.neg_t5_ids = neg_t5_ids
        self.neg_t5_weights = neg_t5_weights
        self.pos_raw = pos_raw
        self.pos_t5_ids = pos_t5_ids
        self.pos_t5_weights = pos_t5_weights
        # Quality-negative baseline for the steering axis (delta = proj_tag -
        # proj_qneg). Decoupled from the CFG negative, which keeps its own role
        # as the uncond base projection. `_qneg_is_neg` is True when the caller
        # left quality_neg empty and we reuse the CFG negative tensors — then the
        # delta is bit-for-bit identical to the pre-decoupling behavior.
        self.qneg_raw = qneg_raw
        self.qneg_t5_ids = qneg_t5_ids
        self.qneg_t5_weights = qneg_t5_weights
        self._qneg_is_neg = qneg_raw is neg_raw
        self.dit = dit
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.taper = int(taper)
        self.taper_scale = float(taper_scale)
        self.final_w = float(final_w)
        # Computed lazily on first forward when DiT is on GPU
        self.cond_combined: Optional[torch.Tensor] = None  # (1, C) base proj_pos
        self.uncond_combined: Optional[torch.Tensor] = None  # (1, C) base proj_neg
        self.delta_unit: Optional[torch.Tensor] = None  # (1, C) proj_tag - proj_neg
        self.per_block_schedule: Optional[List[float]] = None  # length num_blocks
        # σ-FiLM: when the adapter carries a FiLM generator, the projections are
        # timestep-dependent and cannot be precomputed once — the wrapper
        # recomputes them per step from these cached pooled vectors + state.
        self.has_film: bool = False
        self.adapter_state: Optional[dict] = None  # typed, on device
        self.tag_pooled: Optional[torch.Tensor] = None  # (1, pooled_dim)
        self.neg_pooled: Optional[torch.Tensor] = None
        self.pos_pooled: Optional[torch.Tensor] = None
        self.qneg_pooled: Optional[torch.Tensor] = None  # quality-neg baseline

    def _encode_pool(self, raw, t5_ids, t5_weights, device, dtype):
        # The DiT's llm_adapter can live on a different device than the sampling
        # `device` — comfy keeps it CPU-resident here even when the main blocks
        # are GPU-loaded, so feeding cuda ids to a CPU embedding throws an
        # index_select device mismatch. Run the adapter on its own parameter
        # device, then return the pooled result on `device` so the projection
        # MLP (loaded on `device`) matches.
        adapter_device = next(self.dit.llm_adapter.parameters()).device
        adapted = self.dit.preprocess_text_embeds(
            raw.unsqueeze(0).to(device=adapter_device, dtype=dtype),
            t5_ids.unsqueeze(0).to(device=adapter_device)
            if t5_ids is not None
            else None,
            t5xxl_weights=(
                t5_weights.unsqueeze(0)
                .unsqueeze(-1)
                .to(device=adapter_device, dtype=dtype)
                if t5_weights is not None
                else None
            ),
        )
        return adapted.max(dim=1).values.to(device)  # (1, pooled_dim)

    def ensure_precomputed(self, device, dtype):
        """Run LLM adapter + projection for pos / neg / tag once, cache base
        projections and unit steering delta. Schedule built off live block count."""
        if self.cond_combined is not None:
            return

        adapter_state = _get_adapter_typed(self.adapter_path, device, dtype)
        with torch.no_grad():
            tag_pooled = self._encode_pool(
                self.tag_raw, self.tag_t5_ids, self.tag_t5_weights, device, dtype
            )
            neg_pooled = self._encode_pool(
                self.neg_raw, self.neg_t5_ids, self.neg_t5_weights, device, dtype
            )
            pos_pooled = self._encode_pool(
                self.pos_raw, self.pos_t5_ids, self.pos_t5_weights, device, dtype
            )
            # Quality-negative pool feeds ONLY the steering delta. Reuse the CFG
            # negative pool when quality_neg was left empty (legacy behavior).
            qneg_pooled = (
                neg_pooled
                if self._qneg_is_neg
                else self._encode_pool(
                    self.qneg_raw, self.qneg_t5_ids, self.qneg_t5_weights, device, dtype
                )
            )
            proj_tag = _project(tag_pooled, adapter_state)
            proj_neg = _project(neg_pooled, adapter_state)
            proj_pos = _project(pos_pooled, adapter_state)
            proj_qneg = (
                proj_neg if self._qneg_is_neg else _project(qneg_pooled, adapter_state)
            )
            # Base projections — added uniformly via t_embedding_norm hook,
            # matches the line-1640 training-time injection. uncond stays the CFG
            # negative; the quality axis uses proj_qneg.
            self.cond_combined = proj_pos.detach()  # (1, C)
            self.uncond_combined = proj_neg.detach()  # (1, C)
            # Unit steering direction — scaled per-block by `self.per_block_schedule`.
            # σ-flat fallback value; the wrapper overrides it per step when FiLM.
            self.delta_unit = (proj_tag - proj_qneg).detach()
            # Cache pooled + typed state so the wrapper can recompute σ-conditioned
            # projections each step when the adapter is a σ-FiLM head.
            self.has_film = _adapter_has_film(adapter_state)
            if self.has_film:
                # Run the (tiny ~8M-param) FiLM head in fp32 to match its
                # fp32-trained weights; cache the fp32-typed copy ONCE so the
                # per-step recompute does no weight casting. Output is cast back
                # to the block dtype in the wrapper. (_get_adapter_typed caches
                # per (path, device, dtype), so this is a one-time upcast.)
                self.adapter_state = _get_adapter_typed(
                    self.adapter_path, device, torch.float32
                )
                self.tag_pooled = tag_pooled.detach()
                self.neg_pooled = neg_pooled.detach()
                self.pos_pooled = pos_pooled.detach()
                self.qneg_pooled = (
                    self.neg_pooled if self._qneg_is_neg else qneg_pooled.detach()
                )
        self._build_schedule()
        logger.info(
            f"Mod guidance: precomputed "
            f"(w={self.w}, start={self.start_layer}, end={self.end_layer}, "
            f"taper={self.taper}, taper_scale={self.taper_scale}, "
            f"final_w={self.final_w}, sigma_film={self.has_film})"
        )

    def _build_schedule(self):
        # Shared per-block w(ℓ) profile (single source with the library).
        self.per_block_schedule = build_block_schedule(
            len(self.dit.blocks),
            w=self.w,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            taper=self.taper,
            taper_scale=self.taper_scale,
        )


def _t_emb_forward_hook(module, input, output):
    """Module-singleton forward hook — reads ambient delta from the module itself.

    Registered exactly once per t_embedding_norm instance. Per-sample state is
    passed through the `_anima_mod_delta` attribute that `_mod_apply_wrapper` sets/clears
    around each compiled forward. Keeping the hook identity stable across runs
    lets torch.compile's dynamo cache survive between samples.
    """
    delta = getattr(module, "_anima_mod_delta", None)
    if delta is not None:
        return output + delta
    return output


def _ensure_t_emb_hook(dm) -> None:
    t_norm = dm.t_embedding_norm
    if getattr(t_norm, "_anima_mod_hook_installed", False):
        return
    # Pre-initialize so the attribute always exists; dynamo specializes on
    # `is None` vs tensor. Setting it here once means the None branch only
    # gets traced when mod guidance is not active for this forward.
    t_norm._anima_mod_delta = None
    t_norm.register_forward_hook(_t_emb_forward_hook)
    t_norm._anima_mod_hook_installed = True


def _ensure_block_hooks(dm) -> None:
    """Install pre-forward hooks on each block + final_layer that apply
    `w(ℓ) * delta_unit` to the t_embedding argument. Hooks read state from
    `dm._anima_mod_block_state`; when None they fast-no-op."""
    if getattr(dm, "_anima_mod_block_hooks_installed", False):
        return

    dm_ref = weakref.ref(dm)

    def _make_block_prehook(idx: int):
        def _prehook(module, args, kwargs):
            owner = dm_ref()
            if owner is None:
                return None
            st = getattr(owner, "_anima_mod_block_state", None)
            if st is None or st.schedule is None:
                return None
            if idx >= len(st.schedule):
                return None
            w_l = st.schedule[idx]
            if w_l == 0.0:
                return None
            if len(args) < 2:
                return None
            t_emb = args[1]
            delta = st.delta_unit
            if delta.device != t_emb.device or delta.dtype != t_emb.dtype:
                delta = delta.to(device=t_emb.device, dtype=t_emb.dtype)
                st.delta_unit = delta
            new_t_emb = t_emb + (w_l * delta).unsqueeze(1)
            new_args = (args[0], new_t_emb) + tuple(args[2:])
            return new_args, kwargs

        return _prehook

    def _final_prehook(module, args, kwargs):
        owner = dm_ref()
        if owner is None:
            return None
        st = getattr(owner, "_anima_mod_block_state", None)
        if st is None or st.final_w == 0.0:
            return None
        if len(args) < 2:
            return None
        t_emb = args[1]
        delta = st.delta_unit
        if delta.device != t_emb.device or delta.dtype != t_emb.dtype:
            delta = delta.to(device=t_emb.device, dtype=t_emb.dtype)
            st.delta_unit = delta
        new_t_emb = t_emb + (st.final_w * delta).unsqueeze(1)
        new_args = (args[0], new_t_emb) + tuple(args[2:])
        return new_args, kwargs

    for idx, blk in enumerate(dm.blocks):
        blk.register_forward_pre_hook(_make_block_prehook(idx), with_kwargs=True)
    dm.final_layer.register_forward_pre_hook(_final_prehook, with_kwargs=True)
    dm._anima_mod_block_state = None
    dm._anima_mod_block_hooks_installed = True


def _mod_apply_wrapper(executor, *args, **kwargs):
    """APPLY_MODEL wrapper: runs outside torch.compile and writes the prebuilt
    combined tensor to `t_embedding_norm._anima_mod_delta` before the compiled
    DiT forward fires.

    Positional args (from `BaseModel.apply_model`):
        (x, t, c_concat, c_crossattn, control, transformer_options, **extra_conds)
    """
    transformer_options = (
        args[5]
        if len(args) > 5 and isinstance(args[5], dict)
        else kwargs.get("transformer_options", {})
    )
    mod_state = transformer_options.get(MOD_STATE_KEY)
    if mod_state is None:
        # The wrapper is only ever installed by setup_mod_guidance, so reaching
        # here means MOD_STATE_KEY was dropped from transformer_options between
        # setup and sampling — mod guidance silently no-ops. Warn once.
        owner = getattr(executor, "class_obj", None)
        if owner is not None and not getattr(
            owner, "_anima_mod_state_missing_warned", False
        ):
            logger.warning(
                "[mod-guidance] wrapper fired but MOD_STATE_KEY missing from "
                "transformer_options -> mod guidance is a NO-OP this run. "
                "transformer_options keys: %s",
                sorted(k for k in transformer_options.keys() if isinstance(k, str)),
            )
            owner._anima_mod_state_missing_warned = True
        return executor(*args, **kwargs)

    x = args[0]
    device = x.device
    model = executor.class_obj  # BaseModel
    dtype = model.get_dtype_inference()

    # Bind to the LIVE diffusion_model — the module that actually runs this
    # forward — not the ref captured at setup. ComfyUI can hand the sampler a
    # different DiT instance than the one AnimaModGuidance patched upstream (e.g.
    # the model gets re-instantiated between the patcher node and the sampler),
    # which strands the hooks + steering state on a dead module and silently
    # no-ops mod guidance. Re-home here and install the hooks on the live module.
    dit = getattr(model, "diffusion_model", mod_state.dit)
    if dit is None:
        return executor(*args, **kwargs)
    if mod_state.dit is not dit:
        logger.info(
            "[mod-guidance] re-homing to live diffusion_model "
            "(captured id=%x != live id=%x); installing hooks on live module.",
            id(mod_state.dit) & 0xFFFFFF,
            id(dit) & 0xFFFFFF,
        )
        mod_state.dit = dit
        mod_state.cond_combined = None  # invalidate precompute against stale module
    # Idempotent: no-ops once the live module is hooked.
    _ensure_t_emb_hook(dit)
    _ensure_block_hooks(dit)

    # Lazy one-shot: run LLM adapter + projection for pos/neg/tag and build
    # the two (1, C) combined tensors. Subsequent calls early-out.
    mod_state.ensure_precomputed(device, dtype)

    cond_or_uncond = transformer_options.get("cond_or_uncond", [0])

    # σ-FiLM heads are timestep-dependent: recompute the base projections and
    # the steering delta for THIS step's σ. All of this runs in the wrapper
    # (outside torch.compile), so the compiled t_emb / block hooks still only
    # read a finished tensor — the hot path is byte-identical to the σ-flat case.
    combined = None
    delta_unit_typed = None
    if mod_state.has_film and mod_state.adapter_state is not None:
        try:
            t_emb = _compute_t_emb(dit, args[1].to(device))  # (Bt, C)
            state = mod_state.adapter_state  # fp32 (see ensure_precomputed)
            pos_p = mod_state.pos_pooled.to(device)
            neg_p = mod_state.neg_pooled.to(device)
            tag_p = mod_state.tag_pooled.to(device)
            qneg_p = mod_state.qneg_pooled.to(device)
            pieces = []
            for i, cou in enumerate(cond_or_uncond):
                ti = t_emb[i : i + 1] if t_emb.shape[0] > i else t_emb[:1]
                # uncond base row stays the CFG negative.
                pooled_i = pos_p if cou == 0 else neg_p
                # fp32 compute, cast to block dtype on the way out.
                pieces.append(_project_film(pooled_i, state, ti, dtype))
            combined = torch.cat(pieces, dim=0)  # (B, C)
            # Steering delta at this σ (cond row; CFG cond/uncond share σ). Uses
            # the decoupled quality-negative baseline, not the CFG negative.
            t0 = t_emb[:1]
            delta_unit_typed = _project_film(tag_p, state, t0, dtype) - _project_film(
                qneg_p, state, t0, dtype
            )
        except Exception as e:  # graceful: degrade to σ-flat, warn once
            combined = None
            if not getattr(dit, "_anima_film_fallback_warned", False):
                try:
                    tedt = next(dit.t_embedder.parameters()).dtype
                except Exception:
                    tedt = "?"
                adt = (
                    mod_state.adapter_state["0.weight"].dtype
                    if mod_state.adapter_state is not None
                    else "?"
                )
                logger.warning(
                    "[mod-guidance] σ-FiLM recompute fell back to σ-flat (v5 fp32-head): "
                    "%s [t=%s, t_embedder=%s, adapter=%s]. If you just edited the node, "
                    "RESTART the ComfyUI server — re-queuing does NOT reload Python modules.",
                    e,
                    args[1].dtype,
                    tedt,
                    adt,
                )
                dit._anima_film_fallback_warned = True

    if combined is None:
        # σ-flat path: plain adapters, and the FiLM fallback above.
        cond_c = mod_state.cond_combined.to(device=device, dtype=dtype)
        uncond_c = mod_state.uncond_combined.to(device=device, dtype=dtype)
        pieces = [cond_c if cou == 0 else uncond_c for cou in cond_or_uncond]
        combined = torch.cat(pieces, dim=0)  # (B, C)
        delta_unit_typed = mod_state.delta_unit.to(device=device, dtype=dtype)

    # Expose for Spectrum fast-forward (eager, cached steps) and for the
    # t_emb hook (compiled path). Both writes are outside torch.compile.
    dit._mod_pooled_proj = combined.detach()
    t_norm = dit.t_embedding_norm
    t_norm._anima_mod_delta = combined.unsqueeze(1)
    dit._anima_mod_block_state = _PerBlockState(
        delta_unit=delta_unit_typed,
        schedule=mod_state.per_block_schedule,
        final_w=mod_state.final_w,
    )
    try:
        return executor(*args, **kwargs)
    finally:
        t_norm._anima_mod_delta = None
        dit._anima_mod_block_state = None


def _extract_raw_and_t5(conditioning):
    """Pull raw text-encoder output + optional t5 IDs/weights from the first
    entry of a CONDITIONING list and return them as CPU tensors.
    """
    cond_tensor = conditioning[0][0]  # (1, seq, dim) or (seq, dim)
    meta = conditioning[0][1]
    raw = (
        cond_tensor[0].detach().cpu()
        if cond_tensor.ndim == 3
        else cond_tensor.detach().cpu()
    )
    t5_ids = meta.get("t5xxl_ids")
    if t5_ids is not None:
        t5_ids = t5_ids.detach().cpu()
    t5_weights = meta.get("t5xxl_weights")
    if t5_weights is not None:
        t5_weights = t5_weights.detach().cpu()
    return raw, t5_ids, t5_weights


def setup_mod_guidance(
    model_clone,
    clip,
    positive,
    negative,
    adapter_name,
    quality_tags,
    w,
    *,
    quality_neg: str = "",
    start_layer: int = 0,
    end_layer: int = 27,
    taper: int = 0,
    taper_scale: float = 0.25,
    final_w: float = 0.0,
):
    """Capture raw tensors for quality tags / positive / negative, install the
    t_emb / per-block / final-layer hooks, and register the APPLY_MODEL wrapper.

    Called from the KSampler node's sample() before sampling starts. The LLM
    adapter and projection run lazily on the first compiled forward (when the
    DiT is on GPU) and produce the cached `cond_combined` / `uncond_combined`
    base projections plus `delta_unit` steering direction stored on
    `ModGuidanceState`.

    quality_neg gives the steering axis its own negative baseline (delta =
    proj(quality_tags) - proj(quality_neg)), decoupled from the CFG negative.
    Empty string falls back to the CFG negative (legacy behavior).

    Schedule params (per-block guidance shape):
        start_layer:  inclusive; first block to receive `w * delta_unit`.
        end_layer:    exclusive; last block + 1. -1 means num_blocks.
        taper:        number of late slots inside [start, end) to scale by `taper_scale`.
        taper_scale:  multiplier on tapered slots (default 0.25).
        final_w:      `w` applied at `final_layer` (default 0 = don't disturb).
    """
    adapter_path = _resolve_adapter_path(adapter_name)

    # Validate adapter against model
    dm = model_clone.model.diffusion_model
    adapter_cpu = _load_adapter_cpu(adapter_path)
    model_channels = getattr(dm, "model_channels", None)
    if model_channels is None:
        raise RuntimeError("Model missing model_channels")
    if adapter_cpu["0.weight"].shape[0] != model_channels:
        raise RuntimeError(
            f"Adapter output dim ({adapter_cpu['0.weight'].shape[0]}) "
            f"!= model_channels ({model_channels})"
        )

    # Encode quality tags via CLIP (same text encoder the sampler uses)
    tokens = clip.tokenize(quality_tags)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    tag_raw = output["cond"][0].detach().cpu()
    tag_t5_ids = output.get("t5xxl_ids")
    if tag_t5_ids is not None:
        tag_t5_ids = tag_t5_ids.detach().cpu()
    tag_t5_weights = output.get("t5xxl_weights")
    if tag_t5_weights is not None:
        tag_t5_weights = tag_t5_weights.detach().cpu()

    # Extract positive (user prompt) and negative raw embeddings from CONDITIONING
    pos_raw, pos_t5_ids, pos_t5_weights = _extract_raw_and_t5(positive)
    neg_raw, neg_t5_ids, neg_t5_weights = _extract_raw_and_t5(negative)

    # Quality-negative baseline for the steering axis. When provided, tokenize it
    # through CLIP exactly like quality_tags so delta = proj(quality_pos) -
    # proj(quality_neg) is a clean quality counter-pole instead of the broad CFG
    # negative (which is anti-correlated with the quality axis — see
    # docs/proposal/mod_guidance_decoupled_negative.md). Empty → reuse the CFG
    # negative tensors, reproducing the pre-decoupling behavior bit-for-bit.
    if quality_neg and quality_neg.strip():
        qtokens = clip.tokenize(quality_neg)
        qout = clip.encode_from_tokens(qtokens, return_pooled=True, return_dict=True)
        qneg_raw = qout["cond"][0].detach().cpu()
        qneg_t5_ids = qout.get("t5xxl_ids")
        if qneg_t5_ids is not None:
            qneg_t5_ids = qneg_t5_ids.detach().cpu()
        qneg_t5_weights = qout.get("t5xxl_weights")
        if qneg_t5_weights is not None:
            qneg_t5_weights = qneg_t5_weights.detach().cpu()
    else:
        qneg_raw, qneg_t5_ids, qneg_t5_weights = neg_raw, neg_t5_ids, neg_t5_weights

    mod_state = ModGuidanceState(
        adapter_path=adapter_path,
        w=w,
        tag_raw=tag_raw,
        tag_t5_ids=tag_t5_ids,
        tag_t5_weights=tag_t5_weights,
        neg_raw=neg_raw,
        neg_t5_ids=neg_t5_ids,
        neg_t5_weights=neg_t5_weights,
        pos_raw=pos_raw,
        pos_t5_ids=pos_t5_ids,
        pos_t5_weights=pos_t5_weights,
        qneg_raw=qneg_raw,
        qneg_t5_ids=qneg_t5_ids,
        qneg_t5_weights=qneg_t5_weights,
        dit=dm,
        start_layer=start_layer,
        end_layer=end_layer,
        taper=taper,
        taper_scale=taper_scale,
        final_w=final_w,
    )

    # Install the t_emb forward hook exactly once per DiT instance. Module-level
    # hook reads its state from `t_embedding_norm._anima_mod_delta`, which is
    # set / cleared by `_mod_apply_wrapper` outside the compile boundary.
    _ensure_t_emb_hook(dm)
    # Install per-block + final-layer pre-hooks for scheduled steering delta.
    _ensure_block_hooks(dm)

    # Clean up any legacy DIFFUSION_MODEL wrapper from previous versions
    model_clone.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, MOD_DIFFUSION_WRAPPER_KEY
    )

    # Register APPLY_MODEL wrapper — runs outside torch.compile
    model_clone.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL, MOD_APPLY_WRAPPER_KEY
    )
    opts = model_clone.model_options.setdefault("transformer_options", {})
    opts[MOD_STATE_KEY] = mod_state
    model_clone.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL,
        MOD_APPLY_WRAPPER_KEY,
        _mod_apply_wrapper,
    )
