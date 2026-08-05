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

import math

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
    """f_theta: q(z_{t+1} | m_{t+1}, m_t). Training-only; discarded for online generation.

    Optional phase_dim: if > 0, forward() also takes (phase_next, phase_curr) side inputs
    (e.g. [sin(phase), cos(phase)], phase_dim=2), concatenated alongside m_next/m_curr. This does
    NOT change the motion-state m_t definition used elsewhere (RL env guidance rewards etc.) --
    it's purely an extra conditioning signal for this CVAE experiment. Default phase_dim=0
    preserves the exact old behavior/signature for all existing callers.
    """

    def __init__(self, motion_dim: int, latent_dim: int, hidden_sizes: list[int] = [256, 256], phase_dim: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
        self.net = _mlp(2 * motion_dim + 2 * phase_dim, hidden_sizes, 2 * latent_dim)

    def forward(
        self,
        m_next: torch.Tensor,
        m_curr: torch.Tensor,
        phase_next: torch.Tensor | None = None,
        phase_curr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = [m_next, m_curr]
        if self.phase_dim > 0:
            inputs += [phase_next, phase_curr]
        out = self.net(torch.cat(inputs, dim=-1))
        mu, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -10.0, 2.0)
        return mu, logvar


class MotionDecoder(nn.Module):
    """f_phi: p(m_{t+1} | z_{t+1}, m_t). Frozen and used online (auto-regressively) during RL training.

    Optional phase_dim: see MotionEncoder docstring. phase_curr is the current phase (e.g.
    [sin(phase), cos(phase)]), threaded through by the *caller* (never predicted by the model --
    it advances deterministically like a clock, e.g. phase += 2*pi/T each generation step).
    """

    def __init__(self, motion_dim: int, latent_dim: int, hidden_sizes: list[int] = [256, 256], phase_dim: int = 0):
        super().__init__()
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
        self.net = _mlp(latent_dim + motion_dim + phase_dim, hidden_sizes, motion_dim)

    def forward(self, z: torch.Tensor, m_curr: torch.Tensor, phase_curr: torch.Tensor | None = None) -> torch.Tensor:
        inputs = [z, m_curr]
        if self.phase_dim > 0:
            inputs.append(phase_curr)
        return self.net(torch.cat(inputs, dim=-1))


class CommandConditionedEncoder(nn.Module):
    """f_psi(m_t, c_t) -> (mean, logvar) of z. c_t = (forward_vel, yaw_rate), body-frame.

    Trained by gmp/train_command_encoder_selffeed_cmd2d.py via genuine multi-step auto-regressive
    rollout against the frozen MotionDecoder (decoder weights never update). Used online (frozen)
    here to replace random/unconditioned z sampling for the GMP reference generator, so the
    reference trajectory tracks a given velocity command instead of just wandering.
    """

    def __init__(self, motion_dim: int, cmd_dim: int, latent_dim: int, hidden_sizes: list[int], min_std: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.cmd_dim = cmd_dim
        self.min_logvar = 2 * math.log(min_std)
        self.net = _mlp(motion_dim + cmd_dim, hidden_sizes, 2 * latent_dim)

    def forward(self, m_curr: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([m_curr, c], dim=-1))
        mean, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, self.min_logvar, 2.0)
        return mean, logvar


class MotionCVAE(nn.Module):
    """Full CVAE (encoder + decoder) used for offline training on retargeted reference motions."""

    def __init__(self, motion_dim: int, latent_dim: int = 32, hidden_sizes: list[int] = [256, 256], phase_dim: int = 0):
        super().__init__()
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
        self.encoder = MotionEncoder(motion_dim, latent_dim, hidden_sizes, phase_dim=phase_dim)
        self.decoder = MotionDecoder(motion_dim, latent_dim, hidden_sizes, phase_dim=phase_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        m_curr: torch.Tensor,
        m_next: torch.Tensor,
        phase_curr: torch.Tensor | None = None,
        phase_next: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced training step: encode (m_curr, m_next) -> z -> decode -> m_hat_next."""
        mu, logvar = self.encoder(m_next, m_curr, phase_next, phase_curr)
        z = self.reparameterize(mu, logvar)
        m_hat_next = self.decoder(z, m_curr, phase_curr)
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
