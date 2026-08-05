# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Command encoder v6: trains f_psi(m_t, c_t) -> z via deep RL (PPO), instead of the self-feed
multi-step BPTT approach used by train_command_encoder_selffeed_cmd2d.py (kept untouched --
this is a separate, from-scratch alternative, not a replacement).

Follows "Physics-based Character Controllers Using Conditional VAEs" (Won et al., SIGGRAPH
2022), Section 4 / Figure 3a: a Task Encoder is trained by RL to output a latent z which is fed
through a FROZEN, pre-trained motor-decoder; here the "motor-decoder" is our frozen
MotionDecoder and the "Task Encoder" is the command_encoder. No helper branch (Figure 3b) --
per the paper's Figure 3a structure only.

Why RL instead of self-feed/BPTT: the self-feed encoder learned excellent gait quality but was
found (2026-08-05) to barely use forward_vel at all -- even a one-step test on real m_t showed
achieved-speed ~constant (~0.366, the dataset mean) regardless of the commanded value. That
approach trains by imitating a *target z* derived from real data, which the network can satisfy
by simply ignoring c_t. RL instead rewards the *outcome* (did the resulting motion actually
match the commanded speed/yaw), which can't be satisfied by ignoring c_t.

Environment: no physics simulation at all (deliberately -- the GMP reference generator is
already open-loop / policy-independent in g1_gmp_env.py, so training it doesn't need physics
feedback, only the frozen decoder's own m_t -> m_{t+1} transition). This makes rollouts cheap
enough to fully vectorize on GPU.

Exploration-range safeguard (see conversation 2026-08-05): unconstrained RL exploration could
push z outside the region the frozen decoder was actually trained on (its posterior, empirically
~N(0, 0.45) per-dim, not the full N(0,1) prior), producing decoder outputs it's never seen and
was never trained to handle well. Two guards: (a) the policy's std is initialized near that
empirical 0.45 scale and clamped to a ceiling well short of runaway, (b) a small KL-to-prior
regularization term is subtracted from the reward.

Usage:
    python train_command_encoder_rl.py --cvae_checkpoint checkpoints/cvae_sweep240_kl0.01.pt \
        --motion_files ../motions/lafan1_walk/*.npz --output checkpoints/command_encoder_rl.pt
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

    # Table 2 (Task Encoder): MLP, depth 2, width (256, 128), activation (ReLU, ReLU, Linear).
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--value_hidden_sizes", type=int, nargs="+", default=[256, 128])

    # Table 1 (Deep RL, DD-PPO).
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

    # exploration-range safeguard (see module docstring).
    parser.add_argument(
        "--std_init",
        type=float,
        default=0.45,
        help="Initial policy std, grounded in the CVAE's empirically measured aggregate "
        "posterior std (~0.45/dim) rather than either the collapsed self-feed floor (0.085) "
        "or an unconstrained default.",
    )
    parser.add_argument(
        "--std_max",
        type=float,
        default=1.0,
        help="Ceiling on policy std -- keeps exploration within roughly the CVAE prior's own "
        "scale (N(0,1)), short of runaway.",
    )
    parser.add_argument(
        "--kl_prior_weight",
        type=float,
        default=0.01,
        help="Small per-step reward penalty on KL(policy_dist || N(0,1)), discouraging z from "
        "drifting into territory the frozen decoder never saw during its own training.",
    )

    # reward shaping (adapted from the paper's Joystick Control reward, Sec 5.2 -- direction and
    # speed matching as separate exp() terms; our command has no separate lateral component, so
    # this reduces to two independent scalar-tracking terms, matching the existing task reward
    # form already used in g1_gmp_env.py's rew_task_lin_vel/rew_task_ang_vel).
    parser.add_argument("--fwd_temp", type=float, default=5.0)
    parser.add_argument("--yaw_temp", type=float, default=5.0)

    parser.add_argument("--command_fwd_range", type=float, nargs=2, default=[-0.3, 1.2])
    parser.add_argument("--command_yaw_range", type=float, nargs=2, default=[-1.5, 1.5])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class GaussianPolicy(nn.Module):
    """f_psi(m_t, c_t) -> Normal(mean, std) over z. Table 2 architecture (depth 2, width 256/128)."""

    def __init__(self, motion_dim: int, cmd_dim: int, latent_dim: int, hidden_sizes: list[int], std_init: float, std_max: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_logstd = math.log(1e-3)  # generous floor -- only std_max is actively enforced
        self.max_logstd = math.log(std_max)
        self.net = _mlp(motion_dim + cmd_dim, hidden_sizes, 2 * latent_dim)
        # initialize the log_std output channels' bias so the policy starts at std_init instead
        # of an arbitrary default -- see module docstring on why this matters.
        with torch.no_grad():
            last_linear = [m for m in self.net if isinstance(m, nn.Linear)][-1]
            last_linear.bias[latent_dim:].fill_(math.log(std_init))
            last_linear.weight[latent_dim:].mul_(0.01)  # keep logstd near-constant initially

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
    """Per-frame (forward_vel, yaw_rate) in the pelvis's own body frame -- see the identical
    helper in train_command_encoder_selffeed_cmd2d.py for the derivation/rationale."""
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


class CommandEnv:
    """Vectorized, physics-free RL environment: state=m_t, action=z, transition=frozen decoder.

    No world-frame heading is tracked (m_t carries none -- see conversation), so reward uses the
    same body-frame (forward_vel, yaw_rate) decomposition as g1_gmp_env.py's task reward.
    """

    def __init__(self, decoder: MotionDecoder, seed_states: torch.Tensor, args, device):
        self.decoder = decoder
        self.seed_states = seed_states  # (num_seeds, motion_dim), real frames to reset from
        self.args = args
        self.device = device
        self.num_envs = args.num_envs
        self.episode_length = args.episode_length
        self.m = torch.zeros(self.num_envs, seed_states.shape[-1], device=device)
        self.command = torch.zeros(self.num_envs, 2, device=device)
        self.elapsed = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.reset(torch.arange(self.num_envs, device=device))

    def _sample_commands(self, n: int) -> torch.Tensor:
        fwd = torch.empty(n, device=self.device).uniform_(*self.args.command_fwd_range)
        yaw = torch.empty(n, device=self.device).uniform_(*self.args.command_yaw_range)
        return torch.stack([fwd, yaw], dim=-1)

    def reset(self, env_ids: torch.Tensor):
        n = env_ids.shape[0]
        idx = torch.randint(0, self.seed_states.shape[0], (n,), device=self.device)
        self.m[env_ids] = self.seed_states[idx]
        self.command[env_ids] = self._sample_commands(n)
        self.elapsed[env_ids] = 0

    def step(self, z: torch.Tensor, spec: MotionStateSpec):
        with torch.no_grad():
            m_next = self.decoder(z, self.m)
            m_next = torch.clamp(m_next, min=-10.0, max=10.0)
            comp = spec.split(m_next)
            actual_fwd = comp["base_lin_vel"][:, 0]
            actual_yaw = comp["base_ang_vel"][:, 2]

        fwd_err = torch.square(actual_fwd - self.command[:, 0])
        yaw_err = torch.square(actual_yaw - self.command[:, 1])
        reward = torch.exp(-self.args.fwd_temp * fwd_err) * torch.exp(-self.args.yaw_temp * yaw_err)

        self.m = m_next
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

    m_all_list = []
    num_dofs = None
    for motion_file in motion_files:
        loader = MotionLoader(motion_file=motion_file, device=device)
        m_all_list.append(build_motion_state_sequence(loader))
        num_dofs = loader.num_dofs
    seed_states = torch.cat(m_all_list, dim=0)
    spec = MotionStateSpec(num_dofs=num_dofs)
    print(f"[GMP-RL] Loaded {len(motion_files)} clip(s), {seed_states.shape[0]} real frames for episode seeding")

    policy = GaussianPolicy(
        cvae_ckpt["motion_dim"], 2, cvae_ckpt["latent_dim"], args.hidden_sizes, args.std_init, args.std_max
    ).to(device)
    value_fn = ValueNetwork(cvae_ckpt["motion_dim"], 2, args.value_hidden_sizes).to(device)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_fn.parameters()), lr=args.lr)
    prior = torch.distributions.Normal(torch.zeros(cvae_ckpt["latent_dim"], device=device), 1.0)

    env = CommandEnv(decoder, seed_states, args, device)

    rollout_steps = max(1, args.tuples_per_update // args.num_envs)
    print(f"[GMP-RL] num_envs={args.num_envs} rollout_steps={rollout_steps} tuples/update={rollout_steps * args.num_envs}")

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
            # KL-to-prior exploration-range penalty (see module docstring).
            kl = torch.distributions.kl_divergence(dist, prior).sum(-1)
            reward = reward - args.kl_prior_weight * kl

            obs_m[t], obs_c[t] = m_curr, c_curr
            actions[t], logprobs[t] = z, logp
            rewards[t], values[t], dones[t] = reward, v, done.float()

        with torch.no_grad():
            last_value = value_fn(env.m, env.command)

        # GAE (Table 1: gamma=0.99, lambda=0.95).
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
                f"[GMP-RL] update {update:4d}/{args.num_updates} "
                f"mean_reward={rewards.mean().item():.4f} mean_std={mean_std:.4f} "
                f"policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f}"
            )

        # periodic checkpoint -- so progress isn't lost if training is stopped early, and so
        # intermediate checkpoints can be inspected/compared (same practice as the other
        # trainings in this project).
        if update % 20 == 0 or update == args.num_updates - 1:
            root, ext = os.path.splitext(args.output)
            save_checkpoint(f"{root}_update{update}{ext}")
            save_checkpoint(args.output)  # also keep "latest" at the main output path

    print(f"[GMP-RL] Saved {args.output}")


if __name__ == "__main__":
    main()
