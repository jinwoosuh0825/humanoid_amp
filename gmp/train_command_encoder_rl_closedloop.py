# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Command encoder v7: same RL/PPO setup as train_command_encoder_rl.py, but CLOSED-LOOP instead
of open-loop/autoregressive -- kept as a separate file, train_command_encoder_rl.py untouched.

train_command_encoder_rl.py's CommandEnv is open-loop: self.m is overwritten every step with the
decoder's OWN output (self.m = m_next = decoder(z, self.m)), so a policy that's ever
even slightly off-distribution keeps feeding itself further off-distribution inputs -- exactly
the exposure-bias/compounding-error pattern already diagnosed in the self-feed BPTT encoders
(damped oscillation, L/R asymmetry, window-length sensitivity).

Here, self.m is instead advanced along a REAL motion-capture trajectory every step, regardless
of what z the policy picked -- the policy's action still gets rewarded via decoder(z, self.m),
but never gets to see the consequences of its own prediction error, because the next input is
always ground truth. This closes the loop with real state instead of self-generated state.

Caveat (expected going in): in actual deployment (g1_gmp_env.py), the GMP reference generator
still has no access to "real" state -- it IS the open-loop self-feed process this is trying to
avoid learning from. So a policy trained purely closed-loop has never seen its own drifting
output during training, and may not degrade gracefully when finally run open-loop at inference
time. This is a deliberate quick test of whether closed-loop training alone yields better
command-following, not a claim that it solves deployment-time periodicity.

Usage:
    python train_command_encoder_rl_closedloop.py --cvae_checkpoint checkpoints/cvae_sweep240_kl0.01.pt \
        --motion_files ../motions/lafan1_walk/*.npz --output checkpoints/command_encoder_rl_closedloop.pt
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions"))
from motion_loader import MotionLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvae_model import MotionDecoder, _mlp  # noqa: E402
from motion_state import MotionStateSpec, REFERENCE_BODY_NAME, build_motion_state_sequence  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae_checkpoint", type=str, required=True)
    parser.add_argument("--motion_files", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--value_hidden_sizes", type=int, nargs="+", default=[256, 128])

    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--tuples_per_update", type=int, default=50_000)
    parser.add_argument("--epochs_per_update", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip_ratio", type=float, default=0.2)

    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--episode_length", type=int, default=300)
    parser.add_argument("--num_updates", type=int, default=200)

    parser.add_argument("--std_init", type=float, default=0.45)
    parser.add_argument("--std_max", type=float, default=1.0)
    parser.add_argument("--kl_prior_weight", type=float, default=0.01)

    parser.add_argument("--fwd_temp", type=float, default=5.0)
    parser.add_argument("--yaw_temp", type=float, default=5.0)

    parser.add_argument("--command_fwd_range", type=float, nargs=2, default=[-0.3, 1.2])
    parser.add_argument("--command_yaw_range", type=float, nargs=2, default=[-1.5, 1.5])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class GaussianPolicy(nn.Module):
    def __init__(self, motion_dim: int, cmd_dim: int, latent_dim: int, hidden_sizes: list[int], std_init: float, std_max: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_logstd = math.log(1e-3)
        self.max_logstd = math.log(std_max)
        self.net = _mlp(motion_dim + cmd_dim, hidden_sizes, 2 * latent_dim)
        with torch.no_grad():
            last_linear = [m for m in self.net if isinstance(m, nn.Linear)][-1]
            last_linear.bias[latent_dim:].fill_(math.log(std_init))
            last_linear.weight[latent_dim:].mul_(0.01)

    def forward(self, m: torch.Tensor, c: torch.Tensor):
        out = self.net(torch.cat([m, c], dim=-1))
        mean, logstd = out.chunk(2, dim=-1)
        logstd = torch.clamp(logstd, self.min_logstd, self.max_logstd)
        return mean, logstd

    def distribution(self, m: torch.Tensor, c: torch.Tensor) -> torch.distributions.Normal:
        mean, logstd = self.forward(m, c)
        return torch.distributions.Normal(mean, logstd.exp())


class ValueNetwork(nn.Module):
    def __init__(self, motion_dim: int, cmd_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.net = _mlp(motion_dim + cmd_dim, hidden_sizes, 1)

    def forward(self, m: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([m, c], dim=-1)).squeeze(-1)


def compute_body_frame_velocity(loader: MotionLoader) -> torch.Tensor:
    ref_idx = loader.get_body_index([REFERENCE_BODY_NAME])[0]
    quat = loader.body_rotations[:, ref_idx]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    lin_vel = loader.body_linear_velocities[:, ref_idx]
    ang_vel = loader.body_angular_velocities[:, ref_idx]
    vx, vy = lin_vel[:, 0], lin_vel[:, 1]
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    forward_vel = c * vx - s * vy
    yaw_rate = ang_vel[:, 2]
    return torch.stack([forward_vel, yaw_rate], dim=-1)


class ClosedLoopCommandEnv:
    """Same action/reward as CommandEnv, but self.m is advanced along a REAL trajectory each
    step (self.m = real_clip[clip_idx, frame_idx+1]) instead of the decoder's own output. The
    policy's z choice affects the reward (via decoder(z, self.m)) but NOT the next observation
    -- this closes the loop with ground truth instead of self-generated state.
    """

    def __init__(self, decoder: MotionDecoder, clip_tensor: torch.Tensor, clip_lengths: torch.Tensor, args, device):
        self.decoder = decoder
        self.clip_tensor = clip_tensor  # (num_clips, max_len, motion_dim), zero-padded
        self.clip_lengths = clip_lengths  # (num_clips,)
        self.args = args
        self.device = device
        self.num_envs = args.num_envs
        self.episode_length = args.episode_length
        motion_dim = clip_tensor.shape[-1]

        # only clips with enough real frames to seed a full episode are eligible.
        valid = (clip_lengths > args.episode_length + 1).nonzero(as_tuple=True)[0]
        assert valid.numel() > 0, "no motion clip is long enough for the requested episode_length"
        self.valid_clips = valid

        self.m = torch.zeros(self.num_envs, motion_dim, device=device)
        self.command = torch.zeros(self.num_envs, 2, device=device)
        self.clip_idx = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.elapsed = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.reset(torch.arange(self.num_envs, device=device))

    def _sample_commands(self, n: int) -> torch.Tensor:
        fwd = torch.empty(n, device=self.device).uniform_(*self.args.command_fwd_range)
        yaw = torch.empty(n, device=self.device).uniform_(*self.args.command_yaw_range)
        return torch.stack([fwd, yaw], dim=-1)

    def reset(self, env_ids: torch.Tensor):
        n = env_ids.shape[0]
        pick = self.valid_clips[torch.randint(0, self.valid_clips.shape[0], (n,), device=self.device)]
        max_start = (self.clip_lengths[pick] - self.episode_length - 1).clamp(min=0)
        start = (torch.rand(n, device=self.device) * (max_start + 1).float()).long()
        self.clip_idx[env_ids] = pick
        self.frame_idx[env_ids] = start
        self.m[env_ids] = self.clip_tensor[pick, start]
        self.command[env_ids] = self._sample_commands(n)
        self.elapsed[env_ids] = 0

    def step(self, z: torch.Tensor, spec: MotionStateSpec):
        with torch.no_grad():
            m_hat_next = self.decoder(z, self.m)
            m_hat_next = torch.clamp(m_hat_next, min=-10.0, max=10.0)
            comp = spec.split(m_hat_next)
            actual_fwd = comp["base_lin_vel"][:, 0]
            actual_yaw = comp["base_ang_vel"][:, 2]

        fwd_err = torch.square(actual_fwd - self.command[:, 0])
        yaw_err = torch.square(actual_yaw - self.command[:, 1])
        reward = torch.exp(-self.args.fwd_temp * fwd_err) * torch.exp(-self.args.yaw_temp * yaw_err)

        # CLOSED LOOP: next state = real next frame of the trajectory being shadowed, not m_hat_next.
        self.frame_idx += 1
        self.m = self.clip_tensor[self.clip_idx, self.frame_idx]
        self.elapsed += 1
        done = self.elapsed >= self.episode_length
        if done.any():
            self.reset(done.nonzero(as_tuple=True)[0])
        return reward, done


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cvae_ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)
    decoder = MotionDecoder(
        motion_dim=cvae_ckpt["motion_dim"], latent_dim=cvae_ckpt["latent_dim"], hidden_sizes=cvae_ckpt["hidden_sizes"]
    ).to(device)
    decoder.load_state_dict(cvae_ckpt["decoder_state_dict"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    motion_files = []
    for pattern in args.motion_files:
        matches = sorted(glob.glob(pattern))
        motion_files.extend(matches if matches else [pattern])

    clip_list = []
    num_dofs = None
    for motion_file in motion_files:
        loader = MotionLoader(motion_file=motion_file, device=device)
        clip_list.append(build_motion_state_sequence(loader))
        num_dofs = loader.num_dofs
    motion_dim = clip_list[0].shape[-1]
    clip_lengths = torch.tensor([c.shape[0] for c in clip_list], device=device)
    max_len = int(clip_lengths.max().item())
    clip_tensor = torch.zeros(len(clip_list), max_len, motion_dim, device=device)
    for i, c in enumerate(clip_list):
        clip_tensor[i, : c.shape[0]] = c
    spec = MotionStateSpec(num_dofs=num_dofs)
    print(f"[GMP-RL-CL] Loaded {len(motion_files)} clip(s), lengths={clip_lengths.tolist()}")

    policy = GaussianPolicy(
        cvae_ckpt["motion_dim"], 2, cvae_ckpt["latent_dim"], args.hidden_sizes, args.std_init, args.std_max
    ).to(device)
    value_fn = ValueNetwork(cvae_ckpt["motion_dim"], 2, args.value_hidden_sizes).to(device)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_fn.parameters()), lr=args.lr)
    prior = torch.distributions.Normal(torch.zeros(cvae_ckpt["latent_dim"], device=device), 1.0)

    env = ClosedLoopCommandEnv(decoder, clip_tensor, clip_lengths, args, device)

    rollout_steps = max(1, args.tuples_per_update // args.num_envs)
    print(f"[GMP-RL-CL] num_envs={args.num_envs} rollout_steps={rollout_steps} tuples/update={rollout_steps * args.num_envs}")

    def save_checkpoint(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "command_encoder_state_dict": policy.state_dict(),
                "motion_dim": cvae_ckpt["motion_dim"],
                "cmd_dim": 2,
                "latent_dim": cvae_ckpt["latent_dim"],
                "hidden_sizes": args.hidden_sizes,
                "std_max": args.std_max,
                "cvae_checkpoint": args.cvae_checkpoint,
            },
            path,
        )

    for update in range(args.num_updates):
        obs_m = torch.zeros(rollout_steps, args.num_envs, cvae_ckpt["motion_dim"], device=device)
        obs_c = torch.zeros(rollout_steps, args.num_envs, 2, device=device)
        actions = torch.zeros(rollout_steps, args.num_envs, cvae_ckpt["latent_dim"], device=device)
        logprobs = torch.zeros(rollout_steps, args.num_envs, device=device)
        rewards = torch.zeros(rollout_steps, args.num_envs, device=device)
        values = torch.zeros(rollout_steps, args.num_envs, device=device)
        dones = torch.zeros(rollout_steps, args.num_envs, device=device)

        for t in range(rollout_steps):
            m_curr, c_curr = env.m, env.command
            with torch.no_grad():
                dist = policy.distribution(m_curr, c_curr)
                z = dist.sample()
                logp = dist.log_prob(z).sum(-1)
                v = value_fn(m_curr, c_curr)

            reward, done = env.step(z, spec)
            kl = torch.distributions.kl_divergence(dist, prior).sum(-1)
            reward = reward - args.kl_prior_weight * kl

            obs_m[t], obs_c[t] = m_curr, c_curr
            actions[t], logprobs[t] = z, logp
            rewards[t], values[t], dones[t] = reward, v, done.float()

        with torch.no_grad():
            last_value = value_fn(env.m, env.command)

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(args.num_envs, device=device)
        for t in reversed(range(rollout_steps)):
            next_value = last_value if t == rollout_steps - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + args.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values

        b_m = obs_m.reshape(-1, cvae_ckpt["motion_dim"])
        b_c = obs_c.reshape(-1, 2)
        b_actions = actions.reshape(-1, cvae_ckpt["latent_dim"])
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        b_returns = returns.reshape(-1)

        n_tuples = b_m.shape[0]
        for epoch in range(args.epochs_per_update):
            perm = torch.randperm(n_tuples, device=device)
            for start in range(0, n_tuples, args.batch_size):
                idx = perm[start : start + args.batch_size]
                dist = policy.distribution(b_m[idx], b_c[idx])
                new_logp = dist.log_prob(b_actions[idx]).sum(-1)
                ratio = (new_logp - b_logprobs[idx]).exp()

                pg_loss1 = -b_advantages[idx] * ratio
                pg_loss2 = -b_advantages[idx] * torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_pred = value_fn(b_m[idx], b_c[idx])
                value_loss = 0.5 * torch.square(v_pred - b_returns[idx]).mean()

                entropy_bonus = dist.entropy().sum(-1).mean()
                loss = policy_loss + 0.5 * value_loss - 1e-3 * entropy_bonus

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_fn.parameters()), 1.0)
                optimizer.step()

        if update % 5 == 0 or update == args.num_updates - 1:
            mean_std = policy.forward(b_m[:1024], b_c[:1024])[1].exp().mean().item()
            print(
                f"[GMP-RL-CL] update {update:4d}/{args.num_updates} "
                f"mean_reward={rewards.mean().item():.4f} mean_std={mean_std:.4f} "
                f"policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f}"
            )

        if update % 20 == 0 or update == args.num_updates - 1:
            root, ext = os.path.splitext(args.output)
            save_checkpoint(f"{root}_update{update}{ext}")
            save_checkpoint(args.output)

    print(f"[GMP-RL-CL] Saved {args.output}")


if __name__ == "__main__":
    main()
