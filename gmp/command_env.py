# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Lightweight, pure-torch (no Isaac Sim) vectorized environment for training the GMP command
encoder f_psi(c_t, m_t) -> z_{t+1}, following the GMP paper's "Command Encoder" section.

Unlike the main G1 RL environment, "stepping" this environment involves no physics simulation
at all: it just runs the frozen GMP motion decoder forward once. This is what makes training
the command encoder orders of magnitude cheaper than the main locomotion policy.

Motivation: with z sampled purely randomly (z ~ N(0, I)) every step, the auto-regressive
rollout of the decoder was found (see gmp/check_rollout.py) to collapse the generated
base_lin_vel to near-zero within ~20 steps and stay there -- exactly the failure mode the GMP
paper's command encoder is designed to prevent. Training the command encoder to track a
velocity command directly optimizes against this collapse.
"""

from __future__ import annotations

import torch


class CommandTrackingEnv:
    """Vectorized environment: reward = how well the decoded motion tracks a velocity command."""

    def __init__(
        self,
        decoder,
        motion_seed_states: torch.Tensor,
        latent_dim: int,
        horizon: int,
        num_envs: int,
        device: torch.device,
        command_speed_range: tuple[float, float] = (0.1, 0.8),
        command_lateral_range: tuple[float, float] = (-0.1, 0.1),
        reward_temp: float = 2.0,
    ):
        self.decoder = decoder
        self.motion_seed_states = motion_seed_states  # (N, motion_dim), real motion states (for resets)
        self.motion_dim = motion_seed_states.shape[-1]
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = device
        self.command_speed_range = command_speed_range
        self.command_lateral_range = command_lateral_range
        self.reward_temp = reward_temp

        # motion_state.py layout: base_lin_vel occupies the first 3 dims of m_t
        self.m = torch.zeros(num_envs, self.motion_dim, device=device)
        self.c = torch.zeros(num_envs, 3, device=device)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=device)

        self.reset()

    @property
    def obs_dim(self) -> int:
        return 3 + self.motion_dim  # command (3) + motion state

    def _sample_command(self, n: int) -> torch.Tensor:
        vx = torch.empty(n, device=self.device).uniform_(*self.command_speed_range)
        vy = torch.empty(n, device=self.device).uniform_(*self.command_lateral_range)
        vz = torch.zeros(n, device=self.device)
        return torch.stack([vx, vy, vz], dim=-1)

    def _obs(self) -> torch.Tensor:
        return torch.cat([self.c, self.m], dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return self._obs()
        idx = torch.randint(0, self.motion_seed_states.shape[0], (len(env_ids),), device=self.device)
        self.m[env_ids] = self.motion_seed_states[idx]
        self.c[env_ids] = self._sample_command(len(env_ids))
        self.t[env_ids] = 0
        return self._obs()

    def step(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.m = self.decoder(z, self.m)
            # guard against the decoder occasionally diverging over a long auto-regressive
            # rollout (observed with the larger, more diverse 12-clip dataset): real training
            # data's 99.9th percentile |value| is ~3, so this clamp only ever affects runaway
            # values, not normal operation.
            self.m = torch.clamp(self.m, min=-10.0, max=10.0)

        base_lin_vel = self.m[:, 0:3]
        vel_err = torch.norm(base_lin_vel - self.c, dim=-1)
        reward = torch.exp(-self.reward_temp * vel_err)

        self.t += 1
        done = self.t >= self.horizon
        done_ids = done.nonzero(as_tuple=True)[0]
        if len(done_ids) > 0:
            self.reset(done_ids)
        return self._obs(), reward, done
