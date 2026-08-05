# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Command encoder v5: f_psi(m_t, c_t) -> (mean, logvar) of z, extending
train_command_encoder_selffeed.py (which achieved sustained periodic auto-regressive gait with
window_length=200) with an explicit velocity command c_t = (forward_vel, yaw_rate),
expressed in the character's own BODY frame (not world frame).

lateral_vel was dropped after a controllability test on the 3D (forward, lateral, yaw) version
showed forward_vel and yaw_rate both had direction-correct (if compressed) tracking, while
lateral_vel control was unreliable (commanded +0.5 produced achieved -0.11, wrong sign) -- this
matches the real data, where 71% of frames have |lateral_vel| < 0.15 (people rarely sidestep while
walking), so there was little training signal to learn a controllable lateral response from.

Why body frame, not world frame: the real mocap data's world-frame (vx, vy) has near-zero mean in
both components, because the subject continuously turns/changes heading while walking around the
capture volume -- world-frame vy is mostly an artifact of that heading drift, not genuine sideways
motion. Once rotated into the body frame using the REAL stored root orientation (not an integrated,
drift-prone heading), forward_vel is clearly positive-biased (mean~0.36) and lateral_vel is small
and centered near 0 (mean~0.00, std~0.27) -- i.e. people mostly walk forward relative to their own
facing direction, exactly matching the paper's command convention and standard legged-robot control
commands (forward speed, lateral speed, yaw rate).

c_t is derived self-supervised, per training WINDOW (not per-frame, to avoid noisy frame-to-frame
targets and to match "hold a constant commanded velocity for a while" semantics): c_t is fixed to
the window's own average (forward_vel, yaw_rate) for every step of that window's rollout.
Everything else (decoder frozen, m_t scheduled sampling, real multi-step BPTT, window_length=200)
is unchanged from train_command_encoder_selffeed.py.

Usage:
    python train_command_encoder_selffeed_cmd.py --cvae_checkpoint checkpoints/cvae_sweep240_kl0.01.pt \
        --motion_files ../motions/lafan1_walk/*.npz --output checkpoints/command_encoder_selffeed_cmd2d.pt
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
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--window_length", type=int, default=200)
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


class CommandConditionedEncoder(nn.Module):
    """f_psi(m_t, c_t) -> (mean, logvar) of z. c_t = (forward_vel, yaw_rate)."""

    def __init__(self, motion_dim: int, cmd_dim: int, latent_dim: int, hidden_sizes: list[int], min_std: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.cmd_dim = cmd_dim
        self.min_logvar = 2 * math.log(min_std)
        self.net = _mlp(motion_dim + cmd_dim, hidden_sizes, 2 * latent_dim)

    def forward(self, m_curr: torch.Tensor, c: torch.Tensor):
        out = self.net(torch.cat([m_curr, c], dim=-1))
        mean, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, self.min_logvar, 2.0)
        return mean, logvar


def compute_body_frame_velocity(loader: MotionLoader) -> torch.Tensor:
    """Per-frame (forward_vel, yaw_rate) in the pelvis's own body frame, using the REAL stored
    root orientation (not an integrated/drift-prone heading). lateral_vel is dropped -- see
    module docstring."""
    ref_idx = loader.get_body_index([REFERENCE_BODY_NAME])[0]
    quat = loader.body_rotations[:, ref_idx]  # (N, 4) w,x,y,z
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    lin_vel = loader.body_linear_velocities[:, ref_idx]  # world-frame (N, 3)
    ang_vel = loader.body_angular_velocities[:, ref_idx]  # world-frame (N, 3)
    vx, vy = lin_vel[:, 0], lin_vel[:, 1]
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    forward_vel = c * vx - s * vy
    yaw_rate = ang_vel[:, 2]
    return torch.stack([forward_vel, yaw_rate], dim=-1)  # (N, 2)


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
        p.requires_grad_(False)

    motion_files = []
    for pattern in args.motion_files:
        matches = sorted(glob.glob(pattern))
        motion_files.extend(matches if matches else [pattern])

    m_window_list, c_window_list = [], []
    num_dofs, dof_names = None, None
    for motion_file in motion_files:
        loader = MotionLoader(motion_file=motion_file, device=device)
        m = build_motion_state_sequence(loader)
        vel_bf = compute_body_frame_velocity(loader)  # (N, 2)
        num_dofs = loader.num_dofs
        dof_names = loader.dof_names
        m_windows = build_windows(m, args.window_length, args.window_stride)
        vel_windows = build_windows(vel_bf, args.window_length, args.window_stride)
        m_window_list.append(m_windows)
        c_window_list.append(vel_windows.mean(dim=1))  # (num_windows, 2): window-average c_t
    windows = torch.cat(m_window_list, dim=0)
    cmd_targets = torch.cat(c_window_list, dim=0)
    num_windows = windows.shape[0]
    print(f"[GMP] Loaded {len(motion_files)} clip(s), {num_windows} windows of length {args.window_length}")
    print(
        f"[GMP] c_t (window-avg body-frame) stats: "
        f"forward_vel mean/std={cmd_targets[:,0].mean():.3f}/{cmd_targets[:,0].std():.3f}  "
        f"yaw_rate mean/std={cmd_targets[:,1].mean():.3f}/{cmd_targets[:,1].std():.3f}"
    )

    command_encoder = CommandConditionedEncoder(
        ckpt["motion_dim"], 2, ckpt["latent_dim"], args.hidden_sizes, args.min_std
    ).to(device)
    optimizer = torch.optim.Adam(command_encoder.parameters(), lr=args.lr)
    print(f"[GMP] variance floor std={args.min_std:.4f}")

    for epoch in range(args.epochs):
        perm = torch.randperm(num_windows, device=device)
        ss_prob = args.scheduled_sampling_max_prob * (epoch / max(1, args.epochs - 1))
        epoch_loss, num_batches = 0.0, 0
        for start in range(0, num_windows, args.batch_size):
            idx = perm[start : start + args.batch_size]
            batch = windows[idx]
            c_batch = cmd_targets[idx]  # (B, 3), fixed for the whole window

            m_curr = batch[:, 0]
            optimizer.zero_grad()
            total_loss = 0.0
            for t in range(1, args.window_length):
                m_next_true = batch[:, t]
                mean, logvar = command_encoder(m_curr, c_batch)
                std = (0.5 * logvar).exp()
                z = mean + std * torch.randn_like(mean)
                m_hat_next = decoder(z, m_curr)
                step_loss = torch.nn.functional.mse_loss(m_hat_next, m_next_true)
                total_loss = total_loss + step_loss

                if ss_prob > 0.0:
                    mask = (torch.rand(m_curr.shape[0], device=device) < ss_prob).unsqueeze(-1)
                    m_curr = torch.where(mask, m_hat_next, m_next_true)
                else:
                    m_curr = m_next_true

            (total_loss / (args.window_length - 1)).backward()
            optimizer.step()
            epoch_loss += total_loss.item() / (args.window_length - 1)
            num_batches += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[GMP] epoch {epoch:4d}/{args.epochs} mse={epoch_loss / num_batches:.6f} ss_prob={ss_prob:.2f}")

    # rollout check: feed the REAL window-average c_t for the seed segment (self-consistency check)
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
        return float(r[40:120].max()) if len(r) >= 120 else float("nan")

    command_encoder.eval()
    videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
    os.makedirs(videos_dir, exist_ok=True)
    loader0 = MotionLoader(motion_file=motion_files[0], device=device)
    m_all = build_motion_state_sequence(loader0)
    vel_all = compute_body_frame_velocity(loader0)

    torch.manual_seed(0)
    start_frame = 3300
    steps = 300
    c_real = vel_all[start_frame : start_frame + steps].mean(dim=0, keepdim=True)
    m = m_all[start_frame : start_frame + 1].clone()
    knee = np.zeros(steps, dtype=np.float32)
    with torch.no_grad():
        for t in range(steps):
            knee[t] = spec.split(m)["dof_pos"][0, knee_idx].item()
            mean, logvar = command_encoder(m, c_real)
            std = (0.5 * logvar).exp()
            z = mean + std * torch.randn_like(mean)
            m = decoder(z, m)
            m = torch.clamp(m, min=-10.0, max=10.0)
    rebound = autocorr_rebound_detrended(knee)
    print(f"[GMP] rollout (real window-avg c_t): knee_std={knee.std():.3f} detrended_autocorr_rebound={rebound:.3f} c_t={c_real.tolist()}")
    np.save(os.path.join(videos_dir, "knee_command_encoder_selffeed_cmd.npy"), knee)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "command_encoder_state_dict": command_encoder.state_dict(),
            "motion_dim": ckpt["motion_dim"],
            "cmd_dim": 2,
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
