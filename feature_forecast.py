"""Token feature forecasting shared by model-specific Spectrum adapters."""
import logging

import torch

from .spectrum import SpectrumState

logger = logging.getLogger(__name__)


class _Forecaster:
    """Chebyshev (degree ``m``) ridge regression over the step index, blended
    with a first-order Taylor step by ``w``; the last ``k`` observations kept."""

    def __init__(self, m: int, lam: float, w: float, total_steps: int, k: int):
        self.m, self.lam, self.w, self.total, self.k = m, lam, w, total_steps, k
        self.t: list[float] = []
        self.h: list[torch.Tensor] = []

    @property
    def shape(self):
        return self.h[0].shape if self.h else None

    def update(self, t: float, h: torch.Tensor) -> None:
        # Heun and other multi-evaluation solvers revisit a sigma. Keep the
        # latest corrected feature, not duplicate timestamps whose Taylor
        # denominator would collapse to the epsilon guard.
        if self.t and self.t[-1] == float(t):
            self.h[-1] = h.detach()
            return
        self.t.append(float(t))
        self.h.append(h.detach())
        if len(self.t) > self.k:
            del self.t[0], self.h[0]

    def _design(self, t: torch.Tensor) -> torch.Tensor:
        tau = (2.0 * t / self.total - 1.0).reshape(-1, 1)
        cols = [torch.ones_like(tau), tau]
        for _ in range(2, self.m + 1):
            cols.append(2 * tau * cols[-1] - cols[-2])
        return torch.cat(cols[: self.m + 1], 1)

    def weights(self, t_star: float) -> torch.Tensor:
        t = torch.tensor(self.t, dtype=torch.float64)
        x = self._design(t)
        a = x.T @ x + self.lam * torch.eye(x.shape[1], dtype=torch.float64)
        x_star = self._design(torch.tensor([t_star], dtype=torch.float64))[0]
        c = x @ torch.linalg.solve(a, x_star)
        if self.w < 1.0 and len(t) >= 2:
            dt = max(self.t[-1] - self.t[-2], 1e-8)
            kk = (t_star - self.t[-1]) / dt
            taylor = torch.zeros_like(c)
            taylor[-1], taylor[-2] = 1 + kk, -kk
            c = (1 - self.w) * taylor + self.w * c
        return c

    @torch.no_grad()
    def predict(self, t_star: float) -> torch.Tensor:
        c = self.weights(t_star).tolist()
        out = torch.zeros(self.h[0].shape, dtype=torch.float32, device=self.h[0].device)
        for ck, hk in zip(c, self.h):
            out.add_(hk, alpha=ck)
        return out.to(self.h[0].dtype)


class TokenSpectrumState(SpectrumState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-sample head inputs of the running actual forward; None outside one.
        self.features: list | None = None

    def _forecaster_ready(self, key) -> bool:
        forecaster = self.forecasters.get(key)
        return forecaster is not None and len(forecaster.t) >= max(2, self.m_param + 2)


def _update_forecasters(state: TokenSpectrumState, feat, input_x, keys) -> None:
    chunks = feat.chunk(len(keys), dim=0)
    if any(f.shape != chunks[0].shape for f in state.forecasters.values()):
        if state.verbose:
            logger.info("Spectrum: feature shape changed; resetting forecasters")
        state.clear_forecasters()
    for key, h in zip(keys, chunks):
        forecaster = state.forecasters.get(key)
        if forecaster is None:
            forecaster = state.forecasters[key] = _Forecaster(
                state.m_param, state.lam, state.w, state.num_steps, state.history_size
            )
        forecaster.update(float(state.step_idx), h)
    state.record_context(input_x, keys)
