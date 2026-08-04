# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Command encoder v3: z_t = f_psi(m_t, z_{t-1}), stochastic (predicts mean+logvar, samples),
trained with scheduled sampling on z_{t-1}.

Why the non-recurrent version (train_command_encoder_gaussian.py) failed despite mu_psi(m_t)
tracking the true posterior almost perfectly on real data (per-dim correlation 0.999): the
training objective is purely pointwise (per-frame NLL against independently-shuffled examples) --
nothing in it rewards the *resulting z sequence* being temporally coherent when run
auto-regressively, and nothing ever exposes f_psi to a drifted (self-generated, off-real-manifold)
m_t, exactly the OOD-at-generation-time problem the decoder itself has. This version fixes both by
(a) conditioning on z_{t-1} and training on real *ordered* windows so consecutive predictions must
actually cohere, and (b) scheduled sampling on z_{t-1} so f_psi is exposed to its own imperfect
past predictions during training, not just ground truth.

Safeguards (per review, to avoid repeating prior failure modes):
  1. Stochastic by construction (mean+logvar, sampled) -- a deterministic recurrent map is just as
     prone to fixed-point collapse as any other deterministic auto-regressive map we've tested.
  2. Variance floor grounded in the measured real consecutive-step posterior-z distance (~0.18
     average L2 over 32 dims measured earlier), not an arbitrary constant.
  3. Scheduled sampling on z_{t-1} (ramped, like m_curr's scheduled sampling in train_cvae.py) so
     training exposure matches generation-time conditions.

Usage:
    python train_command_encoder_recurrent.py --cvae_checkpoint checkpoints/cvae_sweep240_kl0.01.pt \
        --motion_files ../motions/lafan1_walk/*.npz --output checkpoints/command_encoder_recurrent.pt
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
from cvae_model import MotionEncoder, MotionDecoder, _mlp  # noqa: E402
from motion_state import MotionStateSpec, build_motion_state_sequence  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae_checkpoint", type=str, required=True)
    parser.add_argument("--motion_files", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--window_length", type=int, default=20)
    parser.add_argument("--window_stride", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduled_sampling_max_prob", type=float, default=0.7)
    parser.add_argument(
        "--min_std",
        type=float,
        default=0.18 / (32**0.5),  # ~0.032: derived from the measured real posterior
        # consecutive-step L2 distance (~0.18 over 32 dims) -- see docstring, not an arbitrary pick
        help="Variance floor (as std), grounded in measured real posterior-z step-to-step scale.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class RecurrentCommandEncoder(nn.Module):
    def __init__(self, motion_dim: int, latent_dim: int, hidden_sizes: list[int], min_std: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_logvar = 2 * math.log(min_std)
        self.net = _mlp(motion_dim + latent_dim, hidden_sizes, 2 * latent_dim)

    def forward(self, m_curr: torch.Tensor, z_prev: torch.Tensor):
        out = self.net(torch.cat([m_curr, z_prev], dim=-1))
        mean, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, self.min_logvar, 2.0)
        return mean, logvar


def build_windows(x: torch.Tensor, window_length: int, stride: int) -> torch.Tensor:
    num_frames = x.shape[0]
    if num_frames < window_length:
        return torch.empty(0, window_length, x.shape[-1], device=x.device, dtype=x.dtype)
    starts = list(range(0, num_frames - window_length + 1, stride))
    return torch.stack([x[s : s + window_length] for s in starts], dim=0)


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    var = logvar.exp()
    return torch.mean(torch.sum(0.5 * ((target - mean) ** 2 / var + logvar), dim=-1))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)
    encoder = MotionEncoder(motion_dim=ckpt["motion_dim"], latent_dim=ckpt["latent_dim"], hidden_sizes=ckpt["hidden_sizes"]).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    decoder = MotionDecoder(motion_dim=ckpt["motion_dim"], latent_dim=ckpt["latent_dim"], hidden_sizes=ckpt["hidden_sizes"]).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()
    for p in list(encoder.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)

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

    # precompute the real posterior z (mean) at every position of every window, using the frozen
    # CVAE encoder -- these are the per-step training TARGETS.
    with torch.no_grad():
        z_target_windows = torch.zeros(num_windows, args.window_length - 1, ckpt["latent_dim"], device=device)
        for start in range(0, num_windows, 1024):
            batch = windows[start : start + 1024]
            for t in range(1, args.window_length):
                mu, _ = encoder(batch[:, t], batch[:, t - 1])
                z_target_windows[start : start + batch.shape[0], t - 1] = mu

    command_encoder = RecurrentCommandEncoder(ckpt["motion_dim"], ckpt["latent_dim"], args.hidden_sizes, args.min_std).to(device)
    optimizer = torch.optim.Adam(command_encoder.parameters(), lr=args.lr)
    print(f"[GMP] variance floor std={args.min_std:.4f} (logvar_min={command_encoder.min_logvar:.3f})")

    for epoch in range(args.epochs):
        perm = torch.randperm(num_windows, device=device)
        ss_prob = args.scheduled_sampling_max_prob * (epoch / max(1, args.epochs - 1))
        epoch_loss, num_batches = 0.0, 0
        for start in range(0, num_windows, args.batch_size):
            idx = perm[start : start + args.batch_size]
            batch_m = windows[idx]
            batch_z_target = z_target_windows[idx]

            z_prev = torch.zeros(batch_m.shape[0], ckpt["latent_dim"], device=device)  # z_0 := 0 (prior mean)
            optimizer.zero_grad()
            total_loss = 0.0
            for t in range(1, args.window_length):
                m_curr = batch_m[:, t - 1]  # always real (this experiment isolates z-sequencing, not decoder OOD)
                z_true = batch_z_target[:, t - 1]
                mean, logvar = command_encoder(m_curr, z_prev)
                loss = gaussian_nll(z_true, mean, logvar)
                total_loss = total_loss + loss

                # scheduled sampling on z_{t-1}: expose training to the model's own (sampled, not
                # just mean) past predictions, matching what auto-regressive generation will
                # actually feed back in.
                std = (0.5 * logvar).exp()
                z_sampled = (mean + std * torch.randn_like(mean)).detach()
                if ss_prob > 0.0:
                    mask = (torch.rand(m_curr.shape[0], device=device) < ss_prob).unsqueeze(-1)
                    z_prev = torch.where(mask, z_sampled, z_true)
                else:
                    z_prev = z_true

            (total_loss / (args.window_length - 1)).backward()
            optimizer.step()
            epoch_loss += total_loss.item() / (args.window_length - 1)
            num_batches += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[GMP] epoch {epoch:4d}/{args.epochs} nll={epoch_loss / num_batches:.4f} ss_prob={ss_prob:.2f}")

    # rollout check
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
    m_all = torch.cat([build_motion_state_sequence(MotionLoader(motion_file=motion_files[0], device=device))], dim=0)

    torch.manual_seed(0)
    m = m_all[3300:3301].clone()
    z_prev = torch.zeros(1, ckpt["latent_dim"], device=device)
    steps = 300
    knee = np.zeros(steps, dtype=np.float32)
    with torch.no_grad():
        for t in range(steps):
            knee[t] = spec.split(m)["dof_pos"][0, knee_idx].item()
            mean, logvar = command_encoder(m, z_prev)
            std = (0.5 * logvar).exp()
            z = mean + std * torch.randn_like(mean)
            z_prev = z
            m = decoder(z, m)
            m = torch.clamp(m, min=-10.0, max=10.0)
    rebound = autocorr_rebound_detrended(knee)
    print(f"[GMP] rollout (recurrent, sampled): knee_std={knee.std():.3f} detrended_autocorr_rebound={rebound:.3f}")
    np.save(os.path.join(videos_dir, "knee_command_encoder_recurrent.npy"), knee)

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
