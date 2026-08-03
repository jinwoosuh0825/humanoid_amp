# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Trains the GMP command encoder f_psi(c_t, m_t) -> z_{t+1} with a minimal custom PPO loop,
following the GMP paper's description of training it as a hierarchical policy (latent vector
as the action space) with the motion decoder kept frozen.

No Isaac Sim / skrl involved -- the "environment" (gmp/command_env.py) is just the frozen
decoder, so this trains in minutes on CPU or GPU. See gmp/command_env.py for the motivation.

Usage:
    python train_command_encoder.py --cvae_checkpoint checkpoints/cvae_walk.pt \
        --motion_files ../motions/lafan1_walk/*.npz --output checkpoints/command_encoder.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions"))
from motion_loader import MotionLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvae_model import MotionDecoder, _mlp  # noqa: E402
from motion_state import build_motion_state_sequence  # noqa: E402
from command_env import CommandTrackingEnv  # noqa: E402


class CommandEncoder(nn.Module):
    """f_psi(c_t, m_t) -> z_{t+1}. Gaussian policy: mean network + state-independent log_std."""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_sizes: list[int] = [256, 256, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        self.mean_net = _mlp(obs_dim, hidden_sizes, latent_dim)
        self.log_std = nn.Parameter(torch.zeros(latent_dim) - 1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def act(self, obs: torch.Tensor):
        mean, std = self(obs)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        log_prob = dist.log_prob(z).sum(-1)
        return z, log_prob

    def log_prob(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        mean, std = self(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(z).sum(-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: list[int] = [256, 256, 256]):
        super().__init__()
        self.net = _mlp(obs_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        not_done = (~dones[t]).float()
        delta = rewards[t] + gamma * next_values[t] * not_done - values[t]
        last_adv = delta + gamma * lam * not_done * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae_checkpoint", type=str, required=True)
    parser.add_argument("--motion_files", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--rollout_steps", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning_epochs", type=int, default=4)
    parser.add_argument("--mini_batches", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--ratio_clip", type=float, default=0.2)
    parser.add_argument("--value_loss_scale", type=float, default=0.5)
    parser.add_argument("--entropy_scale", type=float, default=0.0)
    parser.add_argument("--grad_norm_clip", type=float, default=1.0)
    parser.add_argument(
        "--kl_reg_scale",
        type=float,
        default=0.1,
        help="Weight for KL(policy output || N(0,1)) regularization, keeping z within the "
        "decoder's well-trained region instead of only optimizing the velocity reward.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    checkpoint = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)
    decoder = MotionDecoder(
        motion_dim=checkpoint["motion_dim"], latent_dim=checkpoint["latent_dim"], hidden_sizes=checkpoint["hidden_sizes"]
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    motion_files = []
    for pattern in args.motion_files:
        matches = sorted(glob.glob(pattern))
        motion_files.extend(matches if matches else [pattern])

    seed_states_list = []
    for motion_file in motion_files:
        loader = MotionLoader(motion_file=motion_file, device=device)
        seed_states_list.append(build_motion_state_sequence(loader))
    motion_seed_states = torch.cat(seed_states_list, dim=0)
    print(f"[CmdEnc] Loaded {len(motion_files)} motion clip(s), {motion_seed_states.shape[0]} total seed frames")

    env = CommandTrackingEnv(
        decoder=decoder,
        motion_seed_states=motion_seed_states,
        latent_dim=checkpoint["latent_dim"],
        horizon=args.horizon,
        num_envs=args.num_envs,
        device=device,
    )

    policy = CommandEncoder(obs_dim=env.obs_dim, latent_dim=checkpoint["latent_dim"]).to(device)
    value = ValueNet(obs_dim=env.obs_dim).to(device)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=args.learning_rate)

    obs = env.reset()
    batch_size = args.num_envs * args.rollout_steps

    for it in range(args.iterations):
        obs_buf = torch.zeros(args.rollout_steps, args.num_envs, env.obs_dim, device=device)
        z_buf = torch.zeros(args.rollout_steps, args.num_envs, env.latent_dim, device=device)
        logp_buf = torch.zeros(args.rollout_steps, args.num_envs, device=device)
        rew_buf = torch.zeros(args.rollout_steps, args.num_envs, device=device)
        val_buf = torch.zeros(args.rollout_steps, args.num_envs, device=device)
        done_buf = torch.zeros(args.rollout_steps, args.num_envs, dtype=torch.bool, device=device)

        for t in range(args.rollout_steps):
            with torch.no_grad():
                z, log_prob = policy.act(obs)
                v = value(obs)
            next_obs, reward, done = env.step(z)

            obs_buf[t] = obs
            z_buf[t] = z
            logp_buf[t] = log_prob
            rew_buf[t] = reward
            val_buf[t] = v
            done_buf[t] = done
            obs = next_obs

        with torch.no_grad():
            next_values = torch.cat([val_buf[1:], value(obs).unsqueeze(0)], dim=0)
        advantages, returns = compute_gae(rew_buf, val_buf, next_values, done_buf)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_flat = obs_buf.reshape(batch_size, env.obs_dim)
        z_flat = z_buf.reshape(batch_size, env.latent_dim)
        logp_flat = logp_buf.reshape(batch_size)
        adv_flat = advantages.reshape(batch_size)
        ret_flat = returns.reshape(batch_size)

        mb_size = batch_size // args.mini_batches
        for _ in range(args.learning_epochs):
            perm = torch.randperm(batch_size, device=device)
            for mb in range(args.mini_batches):
                idx = perm[mb * mb_size : (mb + 1) * mb_size]
                new_logp = policy.log_prob(obs_flat[idx], z_flat[idx])
                ratio = torch.exp(new_logp - logp_flat[idx])
                surrogate = adv_flat[idx] * ratio
                surrogate_clipped = adv_flat[idx] * torch.clamp(ratio, 1 - args.ratio_clip, 1 + args.ratio_clip)
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                new_v = value(obs_flat[idx])
                value_loss = args.value_loss_scale * torch.nn.functional.mse_loss(new_v, ret_flat[idx])

                # KL(N(mean, std^2) || N(0, 1)) regularization: without this, the velocity-only
                # reward has no reason to keep z close to the prior the decoder was actually
                # trained under (the decoder's coverage of z-space is imperfect -- see
                # gmp/check_rollout.py / cvae_rollout_check.npz), so the policy can wander into
                # an under-trained corner of latent space that happens to also satisfy the
                # velocity target (e.g. a static, non-cycling pose) instead of a natural gait.
                mean, std = policy(obs_flat[idx])
                kl_reg = 0.5 * torch.sum(std.pow(2) + mean.pow(2) - 1.0 - 2 * torch.log(std), dim=-1).mean()

                loss = policy_loss + value_loss + args.kl_reg_scale * kl_reg

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value.parameters()), max_norm=args.grad_norm_clip
                )
                optimizer.step()
                with torch.no_grad():
                    policy.log_std.clamp_(min=-3.0, max=1.0)

        if it % 10 == 0 or it == args.iterations - 1:
            mean_reward = rew_buf.mean().item()
            last_kl_reg = kl_reg.item()
            achieved_vel = env.m[:, 0:3]
            vel_track_err = torch.norm(achieved_vel - env.c, dim=-1).mean().item()
            print(
                f"[CmdEnc] iter {it:4d}/{args.iterations} mean_reward={mean_reward:.4f} "
                f"vel_track_err={vel_track_err:.4f} kl_reg={last_kl_reg:.4f} "
                f"log_std={policy.log_std.mean().item():.3f}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "command_encoder_state_dict": policy.mean_net.state_dict(),
            "obs_dim": env.obs_dim,
            "latent_dim": checkpoint["latent_dim"],
            "hidden_sizes": [256, 256, 256],
            "motion_dim": checkpoint["motion_dim"],
        },
        args.output,
    )
    print(f"[CmdEnc] Saved command encoder checkpoint to {args.output}")


if __name__ == "__main__":
    main()
