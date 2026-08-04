# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Command encoder v4: f_psi(m_t) -> (mean, logvar) of z, trained via genuine multi-step
auto-regressive rollout against the frozen CVAE decoder, with scheduled sampling on m_t itself
(not just on z_prev, which is what the earlier recurrent version got wrong).

Decoder (and its paired encoder) stay strictly frozen: requires_grad_(False), and their
parameters are never added to the optimizer. But the *forward computation* through the decoder
stays part of the autograd graph, so gradient flows back through it to shape command_encoder's
parameters based on losses several steps downstream -- this is NOT joint training (decoder weights
never change), it's genuine multi-step credit assignment for the command_encoder alone.

Why this differs from every prior attempt:
  - train_command_encoder_gaussian.py / _minimal.py: pointwise NLL/MSE against independently
    shuffled (m_t, target_z) pairs -- no sequence-level signal at all, and m_t was always real.
  - train_command_encoder_recurrent.py: added z_{t-1} as input with scheduled sampling on
    z_{t-1}, but m_t was *always real* (`m_curr = batch_m[:, t-1]`) -- so it never tested/trained
    command_encoder's behavior on drifted (self-generated) m_t at all, which independent
    diagnostics pointed to as the actual remaining gap (individual-point mu_psi(m_t) accuracy is
    excellent, ~0.08 L2 error vs a z-scale of ~0.45 -- the problem isn't m_t lacking information).
  - This version: m_t itself is scheduled-sampled (ramped 0 -> max_prob), so command_encoder is
    directly trained and evaluated on exactly the kind of drifted states it will face at real
    generation time, with the loss comparing the resulting closed-loop trajectory to the real one
    at every step (not just NLL against a momentary posterior target).

Usage:
    python train_command_encoder_selffeed.py --cvae_checkpoint checkpoints/cvae_sweep240_kl0.01.pt \
        --motion_files ../motions/lafan1_walk/*.npz --output checkpoints/command_encoder_selffeed.pt
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
from motion_state import MotionStateSpec, build_motion_state_sequence  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae_checkpoint", type=str, required=True)
    parser.add_argument("--motion_files", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--window_length", type=int, default=100, help="~1 real gait cycle.")
    parser.add_argument("--window_stride", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduled_sampling_max_prob", type=float, default=0.9)
    parser.add_argument(
        "--min_std",
        type=float,
        default=0.085,
        help="Variance floor, grounded in the measured individual-level residual (~0.08 L2 err).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class GaussianCommandEncoder(nn.Module):
    def __init__(self, motion_dim: int, latent_dim: int, hidden_sizes: list[int], min_std: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_logvar = 2 * math.log(min_std)
        self.net = _mlp(motion_dim, hidden_sizes, 2 * latent_dim)

    def forward(self, m_curr: torch.Tensor):
        out = self.net(m_curr)
        mean, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, self.min_logvar, 2.0)
        return mean, logvar


def build_windows(x: torch.Tensor, window_length: int, stride: int) -> torch.Tensor:
    num_frames = x.shape[0]
    if num_frames < window_length:
        return torch.empty(0, window_length, x.shape[-1], device=x.device, dtype=x.dtype)
    starts = list(range(0, num_frames - window_length + 1, stride))
    return torch.stack([x[s : s + window_length] for s in starts], dim=0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)
    decoder = MotionDecoder(motion_dim=ckpt["motion_dim"], latent_dim=ckpt["latent_dim"], hidden_sizes=ckpt["hidden_sizes"]).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)  # decoder weights NEVER update -- see module docstring

    motion_files = []
    for pattern in args.motion_files:
        matches = sorted(glob.glob(pattern))
        motion_files.extend(matches if matches else [pattern])

    window_list = []
    num_dofs, dof_names = None, None
    for motion_file in motion_files:
        loader = MotionLoader(motion_file=motion_file, device=device)
        m = build_motion_state_sequence(loader)
        num_dofs = loader.num_dofs
        dof_names = loader.dof_names
        window_list.append(build_windows(m, args.window_length, args.window_stride))
    windows = torch.cat(window_list, dim=0)
    num_windows = windows.shape[0]
    print(f"[GMP] Loaded {len(motion_files)} clip(s), {num_windows} windows of length {args.window_length}")

    command_encoder = GaussianCommandEncoder(ckpt["motion_dim"], ckpt["latent_dim"], args.hidden_sizes, args.min_std).to(device)
    optimizer = torch.optim.Adam(command_encoder.parameters(), lr=args.lr)  # decoder params NOT included
    print(f"[GMP] variance floor std={args.min_std:.4f}")

    for epoch in range(args.epochs):
        perm = torch.randperm(num_windows, device=device)
        ss_prob = args.scheduled_sampling_max_prob * (epoch / max(1, args.epochs - 1))
        epoch_loss, num_batches = 0.0, 0
        for start in range(0, num_windows, args.batch_size):
            idx = perm[start : start + args.batch_size]
            batch = windows[idx]

            m_curr = batch[:, 0]  # only ever real at the very start of the window
            optimizer.zero_grad()
            total_loss = 0.0
            for t in range(1, args.window_length):
                m_next_true = batch[:, t]
                mean, logvar = command_encoder(m_curr)
                std = (0.5 * logvar).exp()
                z = mean + std * torch.randn_like(mean)  # reparameterized, gradient kept (not detached)
                m_hat_next = decoder(z, m_curr)  # decoder frozen, but differentiable -- gradient flows through
                step_loss = torch.nn.functional.mse_loss(m_hat_next, m_next_true)
                total_loss = total_loss + step_loss

                # scheduled sampling on m_t itself (the actual fix vs the recurrent version, which
                # kept m_t always real and only self-fed z_{t-1})
                if ss_prob > 0.0:
                    mask = (torch.rand(m_curr.shape[0], device=device) < ss_prob).unsqueeze(-1)
                    m_curr = torch.where(mask, m_hat_next, m_next_true)  # NOT detached: gradient
                    # must flow across steps for the command_encoder to get real multi-step credit
                else:
                    m_curr = m_next_true

            (total_loss / (args.window_length - 1)).backward()
            optimizer.step()
            epoch_loss += total_loss.item() / (args.window_length - 1)
            num_batches += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[GMP] epoch {epoch:4d}/{args.epochs} mse={epoch_loss / num_batches:.6f} ss_prob={ss_prob:.2f}")

    # rollout check (fully self-fed, matching real generation conditions)
    spec = MotionStateSpec(num_dofs=num_dofs)
    knee_idx = dof_names.index("left_knee_joint") if "left_knee_joint" in dof_names else 0

    def autocorr_rebound_detrended(x, detrend_window=31):
        if len(x) > detrend_window:
            kernel = np.ones(detrend_window) / detrend_window
            trend = np.convolve(x, kernel, mode="same")
            x = x - trend
        x = x - x.mean()
        r = np.correlate(x, x, mode="full")
        r = r[len(r) // 2 :]
        r = r / (r[0] + 1e-8)
        return float(r[80:120].max()) if len(r) >= 120 else float("nan")

    command_encoder.eval()
    videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
    os.makedirs(videos_dir, exist_ok=True)
    m_all = build_motion_state_sequence(MotionLoader(motion_file=motion_files[0], device=device))

    torch.manual_seed(0)
    m = m_all[3300:3301].clone()
    steps = 300
    knee = np.zeros(steps, dtype=np.float32)
    with torch.no_grad():
        for t in range(steps):
            knee[t] = spec.split(m)["dof_pos"][0, knee_idx].item()
            mean, logvar = command_encoder(m)
            std = (0.5 * logvar).exp()
            z = mean + std * torch.randn_like(mean)
            m = decoder(z, m)
            m = torch.clamp(m, min=-10.0, max=10.0)
    rebound = autocorr_rebound_detrended(knee)
    print(f"[GMP] rollout (self-feed trained, sampled): knee_std={knee.std():.3f} detrended_autocorr_rebound={rebound:.3f}")
    np.save(os.path.join(videos_dir, "knee_command_encoder_selffeed.npy"), knee)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "command_encoder_state_dict": command_encoder.state_dict(),
            "motion_dim": ckpt["motion_dim"],
            "latent_dim": ckpt["latent_dim"],
            "hidden_sizes": args.hidden_sizes,
            "min_std": args.min_std,
            "cvae_checkpoint": args.cvae_checkpoint,
        },
        args.output,
    )
    print(f"[GMP] Saved {args.output}")


if __name__ == "__main__":
    main()
