"""ComfyUI disk-cache seam for the SEA δ calibration.

The pure SEA math — the σ-dependent Wiener filter (``sea_filter`` / ``sea_gain`` /
``radial_freq``), the relative-L1 distance (``l1rel``), the auto-δ solver
(``count_refreshes`` / ``solve_delta_for_refresh_ratio``), and the window-schedule
refresh fraction (``window_decision_fraction``) — is the single source of truth in
the Anima repo's ``networks/spectrum_sea.py``, imported here (live repo first,
``_vendor/`` fallback) and re-exported so ``spectrum.py`` keeps using them through
this module unchanged.

This file keeps only the **seam**: persisting the calibrated δ to a ComfyUI-side
JSON cache (user/output dir) keyed by the sampling config. The training repo
persists its own δ under ``output/`` via ``networks/spectrum.py``; the storage
location genuinely differs, so the cache plumbing stays node-side.

SEA filter (SeaCache, Chung et al., arXiv:2602.18993v2, §4.1) — see the vendored
``networks/spectrum_sea.py`` for the full derivation. The δ generalizes across
SMC-CFG / mod-guidance (training-repo bench/spectrum_sea: <0.5% trace
perturbation), which is why the calibration pass can run with mod-guidance active
(this node always does) and the cached δ stays valid.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

# Pure SEA core — single source of truth with the training repo. Re-exported so
# ``spectrum.py`` (and any other node module) can keep importing these from
# ``.spectrum_sea`` directly.
from networks.spectrum_sea import (  # noqa: F401
    accumulate_distances,
    count_refreshes,
    l1rel,
    radial_freq,
    sea_filter,
    sea_gain,
    solve_delta_for_refresh_ratio,
    window_decision_fraction,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# δ disk cache (ComfyUI user/output dir)
# --------------------------------------------------------------------------- #
def _cache_path() -> str:
    """Resolve a persistent path for the δ cache JSON.

    Prefers ComfyUI's user dir (survives node reinstalls, user-discoverable),
    falls back to the output dir, then the node folder for non-ComfyUI contexts.
    """
    try:
        import folder_paths  # type: ignore

        get_user = getattr(folder_paths, "get_user_directory", None)
        base = get_user() if callable(get_user) else folder_paths.get_output_directory()
    except Exception:
        base = os.path.dirname(__file__)
    return os.path.join(base, "spectrum_sea_delta.json")


def make_cache_key(
    num_steps: int,
    warmup_steps: int,
    stop_at: int,
    refresh_ratio: float,
    cfg: float,
    sampler: str,
    h: int,
    w: int,
    extra: str = "",
) -> str:
    """Stable string key. Mirrors the training repo's tuple (prompt deliberately
    excluded — fixed δ + per-prompt-varying refresh pattern is the design).

    ``extra`` carries any non-default substrate that changes the trajectory the
    δ is calibrated against (CFG++ λ, FSG band/K) so its δ never aliases a plain
    run's. Empty (the common case) reproduces the original 8-field key exactly.
    """
    fields = [
        int(num_steps),
        int(warmup_steps),
        int(stop_at),
        round(float(refresh_ratio), 4),
        round(float(cfg), 3),
        str(sampler),
        int(h),
        int(w),
    ]
    if extra:
        fields.append(str(extra))
    return "|".join(str(x) for x in fields)


def load_delta(key: str) -> Optional[float]:
    path = _cache_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
        v = data.get(key)
        return float(v) if v is not None else None
    except Exception as e:  # corrupt/locked cache must never break a generate
        logger.warning("Spectrum SEA: δ cache read failed (%s); recalibrating.", e)
        return None


def save_delta(key: str, value: float) -> None:
    path = _cache_path()
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception:
            data = {}
    data[key] = float(value)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("Spectrum SEA: δ cache write failed (%s); not persisted.", e)
