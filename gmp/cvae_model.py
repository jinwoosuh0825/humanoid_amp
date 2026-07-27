# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Conditional Variational Auto-Encoder (CVAE) motion generator, following the
"Generative Motion Prior" (GMP) design: a motion encoder f_theta(m_t, m_{t+1}) -> z_{t+1}
(training only) and a motion decoder f_phi(z_{t+1}, m_t) -> m_hat_{t+1} (used online, frozen).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    layers = []
    last = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
    layers += [nn.Linear(last, output_dim)]
    return nn.Sequential(*layers)


class MotionEncoder(nn.Module):
    """f_theta: q(z_{t+1} | m_{t+1}, m_t). Training-only; discarded for online generation."""

    def __init__(self, motion_dim: int, latent_dim: int, hidden_sizes: list[int] = [256, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = _mlp(2 * motion_dim, hidden_sizes, 2 * latent_dim)

    def forward(self, m_next: torch.Tensor, m_curr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([m_next, m_curr], dim=-1))
        mu, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -10.0, 2.0)
        return mu, logvar


class MotionDecoder(nn.Module):
    """f_phi: p(m_{t+1} | z_{t+1}, m_t). Frozen and used online (auto-regressively) during RL training."""

    def __init__(self, motion_dim: int, latent_dim: int, hidden_sizes: list[int] = [256, 256]):
        super().__init__()
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.net = _mlp(latent_dim + motion_dim, hidden_sizes, motion_dim)

    def forward(self, z: torch.Tensor, m_curr: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, m_curr], dim=-1))


class MotionCVAE(nn.Module):
    """Full CVAE (encoder + decoder) used for offline training on retargeted reference motions."""

    def __init__(self, motion_dim: int, latent_dim: int = 32, hidden_sizes: list[int] = [256, 256]):
        super().__init__()
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.encoder = MotionEncoder(motion_dim, latent_dim, hidden_sizes)
        self.decoder = MotionDecoder(motion_dim, latent_dim, hidden_sizes)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, m_curr: torch.Tensor, m_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced training step: encode (m_curr, m_next) -> z -> decode -> m_hat_next."""
        mu, logvar = self.encoder(m_next, m_curr)
        z = self.reparameterize(mu, logvar)
        m_hat_next = self.decoder(z, m_curr)
        return m_hat_next, mu, logvar

    @torch.no_grad()
    def generate_next(self, m_curr: torch.Tensor) -> torch.Tensor:
        """Online (frozen) generation: sample z ~ N(0, I) and decode the next motion state.

        No command encoder is used (unconditioned generation) -- suitable for pure motion
        imitation tasks that do not require velocity-command tracking.
        """
        z = torch.randn(*m_curr.shape[:-1], self.latent_dim, device=m_curr.device, dtype=m_curr.dtype)
        return self.decoder(z, m_curr)


def vae_loss(
    m_hat_next: torch.Tensor,
    m_next: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    rec_weight: float = 1.0,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """CVAE loss: reconstruction (MSE) + KL divergence to N(0, I), as in the GMP paper."""
    rec_loss = torch.mean(torch.sum((m_hat_next - m_next) ** 2, dim=-1))
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    total = rec_weight * rec_loss + kl_weight * kl_loss
    return total, {"rec_loss": rec_loss.detach(), "kl_loss": kl_loss.detach()}
