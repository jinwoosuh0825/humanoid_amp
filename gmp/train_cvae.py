# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Offline training script for the GMP motion CVAE.

Trains the motion encoder/decoder on a retargeted reference motion (e.g. motions/G1_walk.npz)
using reconstruction (MSE) + KL-divergence loss, following the GMP paper. Only the decoder is
needed afterwards (used frozen, online, during RL training).

Usage:
    python train_cvae.py --motion_file ../motions/G1_walk.npz --output cvae_walk.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions"))
from motion_loader import MotionLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvae_model import MotionCVAE, vae_loss  # noqa: E402
from motion_state import build_motion_state_sequence  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr_start", type=float, default=1e-5)
    parser.add_argument("--lr_end", type=float, default=1e-7)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--scheduled_sampling_start_epoch", type=int, default=120)
    parser.add_argument("--scheduled_sampling_max_prob", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    loader = MotionLoader(motion_file=args.motion_file, device=device)
    m = build_motion_state_sequence(loader)  # (num_frames, motion_dim)
    motion_dim = m.shape[-1]
    print(f"[GMP] Loaded motion '{args.motion_file}': {m.shape[0]} frames, motion_dim={motion_dim}")

    m_curr_all = m[:-1]
    m_next_all = m[1:]
    num_pairs = m_curr_all.shape[0]

    model = MotionCVAE(motion_dim=motion_dim, latent_dim=args.latent_dim, hidden_sizes=args.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start)
    gamma = (args.lr_end / args.lr_start) ** (1.0 / max(1, args.epochs - 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(args.epochs):
        perm = torch.randperm(num_pairs, device=device)
        # scheduled sampling probability: ramps up linearly after `scheduled_sampling_start_epoch`
        if epoch < args.scheduled_sampling_start_epoch:
            ss_prob = 0.0
        else:
            progress = (epoch - args.scheduled_sampling_start_epoch) / max(
                1, args.epochs - args.scheduled_sampling_start_epoch
            )
            ss_prob = args.scheduled_sampling_max_prob * min(1.0, progress)

        epoch_rec, epoch_kl, num_batches = 0.0, 0.0, 0
        for start in range(0, num_pairs, args.batch_size):
            idx = perm[start : start + args.batch_size]
            m_curr = m_curr_all[idx]
            m_next = m_next_all[idx]

            if ss_prob > 0.0:
                # scheduled sampling: replace the conditioning frame with the model's own
                # (frozen-grad) prediction for a random subset of the batch, so the decoder
                # learns to stay robust to its own auto-regressive drift.
                with torch.no_grad():
                    m_pred = model.generate_next(m_curr)
                mask = (torch.rand(m_curr.shape[0], device=device) < ss_prob).unsqueeze(-1)
                m_curr = torch.where(mask, m_pred, m_curr)

            m_hat_next, mu, logvar = model(m_curr, m_next)
            loss, logs = vae_loss(m_hat_next, m_next, mu, logvar, args.rec_weight, args.kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_rec += logs["rec_loss"].item()
            epoch_kl += logs["kl_loss"].item()
            num_batches += 1

        scheduler.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"[GMP] epoch {epoch:4d}/{args.epochs} "
                f"rec={epoch_rec / num_batches:.6f} kl={epoch_kl / num_batches:.6f} "
                f"ss_prob={ss_prob:.2f} lr={scheduler.get_last_lr()[0]:.2e}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "decoder_state_dict": model.decoder.state_dict(),
            "motion_dim": motion_dim,
            "latent_dim": args.latent_dim,
            "hidden_sizes": args.hidden_sizes,
            "motion_file": args.motion_file,
        },
        args.output,
    )
    print(f"[GMP] Saved frozen decoder checkpoint to {args.output}")


if __name__ == "__main__":
    main()
