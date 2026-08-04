# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Minimal command encoder: f_psi(m_t) -> z_{t+1}, trained by supervised regression against the
*frozen CVAE's own posterior z* (z = encoder(m_{t+1}, m_t)), instead of the paper's full RL
("hierarchical policy") pipeline. No user velocity command c_t -- for this diagnostic, the goal is
just "reproduce plain walking", so command is fixed/omitted.

Why this should work: six prior fixes (kl_weight, prior-z exposure, window_length, real BPTT,
cycle-consistency loss, explicit fixed-T phase) all failed to make auto-regressive rollouts
periodic. A shuffle-vs-posterior contrast test showed posterior z is both (a) informative (~2/3 of
the benefit) and (b) temporally smooth (consecutive-step distance 0.18 vs 7.96 for iid random z).
An EMA-smoothed random z (alpha~0.3-0.5) already recovered a genuine (if modest) periodicity signal
with NO other change. This model directly predicts (an approximation of) that same smooth,
informative posterior z from m_t alone -- combining both benefits by construction, since it's
supervised to match a signal that is already both informative and smooth.

The decoder (and its paired encoder, used only to generate training TARGETS here) are loaded
frozen from an already-trained, non-collapsed checkpoint (e.g. cvae_sweep240_kl0.01.pt) -- no
decoder retraining needed.

Usage:
    python train_command_encoder_minimal.py --cvae_checkpoint checkpoints/cvae_sweep240_kl0.01.pt \
        --motion_files ../motions/lafan1_walk/*.npz --output checkpoints/command_encoder_minimal.pt
"""

from __future__ import annotations

import argparse
import glob
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

    command_encoder = _mlp(ckpt["motion_dim"], args.hidden_sizes, ckpt["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(command_encoder.parameters(), lr=args.lr)

    num_pairs = m_curr_all.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(num_pairs, device=device)
        epoch_loss, num_batches = 0.0, 0
        for start in range(0, num_pairs, args.batch_size):
            idx = perm[start : start + args.batch_size]
            m_curr, m_next = m_curr_all[idx], m_next_all[idx]
            with torch.no_grad():
                z_target, _ = encoder(m_next, m_curr)  # posterior mean, frozen -- the "teacher" z
            z_pred = command_encoder(m_curr)
            loss = torch.nn.functional.mse_loss(z_pred, z_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"[GMP] epoch {epoch:4d}/{args.epochs} z_regression_mse={epoch_loss / num_batches:.6f}")

    # quick rollout check right here (no need for a separate script/round-trip)
    spec = MotionStateSpec(num_dofs=num_dofs)
    knee_idx = dof_names.index("left_knee_joint") if "left_knee_joint" in dof_names else 0

    def autocorr_rebound(x):
        x = x - x.mean()
        r = np.correlate(x, x, mode="full")
        r = r[len(r) // 2 :]
        r = r / (r[0] + 1e-8)
        return float(r[80:120].max()) if len(r) >= 120 else float("nan")

    command_encoder.eval()
    m = m_curr_all[3300:3301].clone()
    steps = 300
    knee = np.zeros(steps, dtype=np.float32)
    z_seq = []
    with torch.no_grad():
        for t in range(steps):
            knee[t] = spec.split(m)["dof_pos"][0, knee_idx].item()
            z = command_encoder(m)
            z_seq.append(z[0].cpu().numpy())
            m = decoder(z, m)
            m = torch.clamp(m, min=-10.0, max=10.0)
    z_seq = np.array(z_seq)
    z_step_dist = np.linalg.norm(z_seq[1:] - z_seq[:-1], axis=-1).mean()
    print(f"[GMP] rollout (deterministic command_encoder): knee_std={knee.std():.3f} "
          f"autocorr_rebound={autocorr_rebound(knee):.3f} z_consecutive_step_dist={z_step_dist:.4f}")
    videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
    os.makedirs(videos_dir, exist_ok=True)
    np.save(os.path.join(videos_dir, "knee_command_encoder_minimal.npy"), knee)

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
