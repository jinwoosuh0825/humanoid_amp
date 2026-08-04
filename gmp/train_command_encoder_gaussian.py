# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Command encoder v2: f_psi(m_t) -> (mean(z), logvar(z)), trained via Gaussian NLL against the
frozen CVAE's posterior z targets, then SAMPLED from at rollout time (not used deterministically).

Why this differs from train_command_encoder_minimal.py (plain MSE regression): MSE regression
against a single target collapses to the conditional mean when m_t doesn't fully determine a
unique target z (i.e. similar m_t across different real cycles/speeds can have somewhat different
"correct" continuations) -- this is a well-known failure mode of point regression under aleatoric
ambiguity, and matches what we observed (the MSE-regression command encoder converged to a clean
fixed point in rollout, worse than doing nothing). A heteroscedastic Gaussian output lets the
network express "I'm genuinely unsure here" via a wider predicted variance instead of averaging
away the ambiguity, and sampling from that predicted distribution (rather than always taking the
mean) injects exactly the kind of state-conditioned, non-degenerate stochasticity needed to avoid
collapsing onto a single fixed point -- while (unlike iid N(0,1) or EMA noise) it's centered per
m_t on what the frozen CVAE encoder's posterior actually looked like for similar states.

Usage:
    python train_command_encoder_gaussian.py --cvae_checkpoint checkpoints/cvae_sweep240_kl0.01.pt \
        --motion_files ../motions/lafan1_walk/*.npz --output checkpoints/command_encoder_gaussian.pt
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class GaussianCommandEncoder(nn.Module):
    def __init__(self, motion_dim: int, latent_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = _mlp(motion_dim, hidden_sizes, 2 * latent_dim)

    def forward(self, m_curr: torch.Tensor):
        out = self.net(m_curr)
        mean, logvar = out.chunk(2, dim=-1)
        # floor the variance well above 0: Gaussian NLL is notorious for collapsing variance
        # toward 0 to cheat the loss (shrinking the log(var) term costs less than actually
        # reducing the squared error), which defeats the whole point of predicting a
        # heteroscedastic distribution -- we saw exactly this (predicted var collapsed to
        # ~0.0003) in the unfloored version, making "sampled" behave identically to "mean_only".
        logvar = torch.clamp(logvar, math.log(0.15**2), 2.0)
        return mean, logvar


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

    m_curr_list, m_next_list = [], []
    num_dofs, dof_names = None, None
    for motion_file in motion_files:
        loader = MotionLoader(motion_file=motion_file, device=device)
        m = build_motion_state_sequence(loader)
        num_dofs = loader.num_dofs
        dof_names = loader.dof_names
        m_curr_list.append(m[:-1])
        m_next_list.append(m[1:])
    m_curr_all = torch.cat(m_curr_list, dim=0)
    m_next_all = torch.cat(m_next_list, dim=0)
    print(f"[GMP] Loaded {len(motion_files)} clip(s), {m_curr_all.shape[0]} (m_t, m_t+1) pairs")

    command_encoder = GaussianCommandEncoder(ckpt["motion_dim"], ckpt["latent_dim"], args.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(command_encoder.parameters(), lr=args.lr)

    num_pairs = m_curr_all.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(num_pairs, device=device)
        epoch_loss, epoch_var, num_batches = 0.0, 0.0, 0
        for start in range(0, num_pairs, args.batch_size):
            idx = perm[start : start + args.batch_size]
            m_curr, m_next = m_curr_all[idx], m_next_all[idx]
            with torch.no_grad():
                z_target, _ = encoder(m_next, m_curr)
            mean, logvar = command_encoder(m_curr)
            loss = gaussian_nll(z_target, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_var += logvar.exp().mean().item()
            num_batches += 1
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"[GMP] epoch {epoch:4d}/{args.epochs} nll={epoch_loss / num_batches:.4f} "
                  f"mean_predicted_var={epoch_var / num_batches:.4f}")

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

    # both deterministic (mean only) and stochastic (sampled) rollouts, for direct comparison
    for mode in ["mean_only", "sampled"]:
        torch.manual_seed(0)
        m = m_curr_all[3300:3301].clone()
        steps = 300
        knee = np.zeros(steps, dtype=np.float32)
        with torch.no_grad():
            for t in range(steps):
                knee[t] = spec.split(m)["dof_pos"][0, knee_idx].item()
                mean, logvar = command_encoder(m)
                if mode == "sampled":
                    std = (0.5 * logvar).exp()
                    z = mean + std * torch.randn_like(mean)
                else:
                    z = mean
                m = decoder(z, m)
                m = torch.clamp(m, min=-10.0, max=10.0)
        rebound = autocorr_rebound_detrended(knee)
        print(f"[GMP] rollout mode={mode}: knee_std={knee.std():.3f} detrended_autocorr_rebound={rebound:.3f}")
        np.save(os.path.join(videos_dir, f"knee_command_encoder_gaussian_{mode}.npy"), knee)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "command_encoder_state_dict": command_encoder.state_dict(),
            "motion_dim": ckpt["motion_dim"],
            "latent_dim": ckpt["latent_dim"],
            "hidden_sizes": args.hidden_sizes,
            "cvae_checkpoint": args.cvae_checkpoint,
        },
        args.output,
    )
    print(f"[GMP] Saved {args.output}")


if __name__ == "__main__":
    main()
