"""Pure tensor utilities for decoupled DMD on the FireRed flow convention.

FireRed uses ``x_t = t * noise + (1 - t) * x0``: ``t=1`` is noisy and
``t=0`` is clean.  The focused CA schedule therefore samples a re-noise time
below the current generator time, unlike the convention used in the D-DMD
paper pseudocode.
"""

from __future__ import annotations

import torch


class DMGradientOverrides:
    """Explicit draws used only by deterministic trainer regression tests."""

    def __init__(
        self,
        *,
        t_dm: torch.Tensor,
        dm_noise: torch.Tensor,
        t_ca: torch.Tensor | None = None,
        ca_noise: torch.Tensor | None = None,
    ) -> None:
        self.t_dm = t_dm
        self.dm_noise = dm_noise
        self.t_ca = t_ca
        self.ca_noise = ca_noise


def expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if t.ndim != 1 or int(t.shape[0]) != int(x.shape[0]):
        raise ValueError(f"t must have shape [batch], got t={tuple(t.shape)} x={tuple(x.shape)}")
    return t.view(t.shape[0], *([1] * (x.dim() - 1)))


def noised_latents_from_noise(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    if x0.shape != noise.shape:
        raise ValueError(f"x0 and noise shapes must match, got {tuple(x0.shape)} and {tuple(noise.shape)}")
    t_expanded = expand_time(t, x0)
    return t_expanded * noise + (1.0 - t_expanded) * x0


def sample_uniform_time(
    *,
    batch_size: int,
    device: torch.device,
    low: float,
    high: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"time bounds must satisfy 0 <= low < high <= 1, got low={low} high={high}")
    return torch.rand(batch_size, device=device, generator=generator).mul(high - low).add(low)


def sample_constrained_ca_time(
    *,
    t_gen: torch.Tensor,
    low: float,
    high: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample focused CA times in ``[low, min(high, t_gen))`` per sample."""
    if t_gen.ndim != 1 or t_gen.numel() == 0:
        raise ValueError(f"t_gen must be a non-empty rank-1 tensor, got {tuple(t_gen.shape)}")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"CA bounds must satisfy 0 <= low < high <= 1, got low={low} high={high}")
    upper = torch.minimum(t_gen.float(), torch.full_like(t_gen.float(), high))
    if not bool(torch.all(upper > low).item()):
        raise ValueError(
            "Focused CA has an empty interval: every t_gen must be greater than ca_noise_t_min; "
            f"low={low} min_t_gen={float(t_gen.min().item())}"
        )
    uniforms = torch.rand(t_gen.shape, device=t_gen.device, dtype=torch.float32, generator=generator)
    return uniforms.mul(upper - low).add(low)


def coupled_raw_gradient(
    *,
    fake_x0: torch.Tensor,
    real_cond_x0: torch.Tensor,
    real_uncond_x0: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Return the legacy DMD2 ``fake - teacher_cfg`` x0-space gradient."""
    real_cfg_x0 = real_uncond_x0 + guidance_scale * (real_cond_x0 - real_uncond_x0)
    return fake_x0 - real_cfg_x0


def decoupled_raw_gradient(
    *,
    fake_dm_x0: torch.Tensor,
    real_cond_dm_x0: torch.Tensor,
    real_cond_ca_x0: torch.Tensor,
    real_uncond_ca_x0: torch.Tensor,
    guidance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return combined, DM, and CA x0-space gradients.

    When DM and CA use the same re-noised latent and time, this is algebraically
    identical to :func:`coupled_raw_gradient`.
    """
    delta_dm = fake_dm_x0 - real_cond_dm_x0
    delta_ca = -(guidance_scale - 1.0) * (real_cond_ca_x0 - real_uncond_ca_x0)
    return delta_dm + delta_ca, delta_dm, delta_ca


def normalize_and_clip_gradient(
    gradient: torch.Tensor,
    *,
    eps: float,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if clip <= 0.0:
        raise ValueError(f"clip must be positive, got {clip}")
    norm = gradient.abs().mean().clamp_min(eps)
    normalized = torch.nan_to_num(
        gradient / norm,
        nan=0.0,
        posinf=clip,
        neginf=-clip,
    ).clamp(-clip, clip)
    return normalized, norm
