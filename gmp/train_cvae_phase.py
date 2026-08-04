# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Phase-conditioned CVAE experiment (Option A: fixed-period phase clock).

Six prior approaches (kl_weight tuning, prior-z exposure, longer window_length, real multi-step
BPTT, explicit cycle-consistency loss) all failed to make auto-regressive random-z rollouts
periodic, despite a recurrence-plot diagnostic showing the real m_t representation itself does
carry phase information along the real trajectory. The working hypothesis: the decoder has to
*re-infer* "where in the gait cycle am I" from m_t alone every single step, and that inference is
an unstable sub-problem for gradient-based optimization to get robust once the state drifts even
slightly off the real trajectory manifold. This script tests whether removing that sub-problem --
by giving the encoder/decoder an explicit (sin(phase), cos(phase)) side-input that advances like a
deterministic clock (phase += 2*pi/T every step, NEVER predicted by the model) -- fixes this.

This does NOT change the motion_state.py `m_t` definition (95-dim) used everywhere else (RL env
guidance rewards, command encoder, etc.) -- phase is purely an extra conditioning input threaded
through cvae_model.py's optional phase_dim mechanism.

Built-in safeguards (per review, to avoid another multi-turn confound debate):
  1. Trains a phase model (phase_dim=2) AND a no-phase baseline (phase_dim=0) with identical
     everything else, and compares teacher-forced reconstruction MSE BEFORE looking at rollout
     quality. If phase actively hurts teacher-forced MSE, that signals the fixed T=100 label is
     inconsistent with the real data (not that "phase as a concept" is a bad idea) -- pointing
     straight at foot-contact-based re-labeling as the next step, without re-litigating this.
  2. Phase ablation, logged alongside z ablation: rollout with (a) the real advancing clock,
     (b) phase frozen at its initial value, (c) phase re-randomized every step -- checks whether
     the decoder actually *uses* phase or just ignores it (the same collapse risk z had).

Usage:
    python train_cvae_phase.py --motion_files ../motions/lafan1_walk/*.npz \
        --output_prefix checkpoints/cvae_phase_T100 --cycle_period 100 --epochs 150
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions"))
from motion_loader import MotionLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvae_model import MotionCVAE, vae_loss  # noqa: E402
from motion_state import MotionStateSpec, build_motion_state_sequence  # noqa: E402

PHASE_DIM = 2  # [sin(phase), cos(phase)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_files", type=str, nargs="+", required=True)
    parser.add_argument("--output_prefix", type=str, required=True, help="Saves {prefix}_phase.pt and {prefix}_nophase.pt")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr_start", type=float, default=1e-5)
    parser.add_argument("--lr_end", type=float, default=1e-7)
    parser.add_argument("--kl_weight", type=float, default=0.05, help="Best non-collapsing value found in the kl sweep.")
    parser.add_argument("--window_length", type=int, default=20)
    parser.add_argument("--window_stride", type=int, default=10)
    parser.add_argument("--scheduled_sampling_max_prob", type=float, default=0.95)
    parser.add_argument("--cycle_period", type=int, default=100, help="Fixed nominal gait-cycle length T, in frames.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_windows(x: torch.Tensor, window_length: int, stride: int) -> torch.Tensor:
    num_frames = x.shape[0]
    if num_frames < window_length:
        return torch.empty(0, window_length, x.shape[-1], device=x.device, dtype=x.dtype)
    starts = list(range(0, num_frames - window_length + 1, stride))
    return torch.stack([x[s : s + window_length] for s in starts], dim=0)


def compute_phase_sincos(num_frames: int, period: int, device) -> torch.Tensor:
    """phase[i] = 2*pi*(i mod period)/period -- a fixed clock from the start of the clip.
    Known limitation (safeguard #1 checks for this): ignores real speed changes/standing segments,
    so it will NOT track the true biomechanical phase during those. Returns (num_frames, 2)."""
    idx = torch.arange(num_frames, dtype=torch.float32, device=device)
    phase = 2 * math.pi * (idx % period) / period
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)


def train_one(model: MotionCVAE, windows_m: torch.Tensor, windows_phase, args, device, tag: str):
    """windows_phase is None for the no-phase baseline (phase_dim=0)."""
    use_phase = windows_phase is not None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start)
    gamma = (args.lr_end / args.lr_start) ** (1.0 / max(1, args.epochs - 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    num_windows = windows_m.shape[0]

    for epoch in range(args.epochs):
        perm = torch.randperm(num_windows, device=device)
        ss_prob = args.scheduled_sampling_max_prob * (epoch / max(1, args.epochs - 1))
        epoch_rec, epoch_kl, num_batches = 0.0, 0.0, 0
        for start in range(0, num_windows, args.batch_size):
            idx = perm[start : start + args.batch_size]
            batch_m = windows_m[idx]
            batch_phase = windows_phase[idx] if use_phase else None

            m_curr = batch_m[:, 0]
            batch_rec, batch_kl = 0.0, 0.0
            optimizer.zero_grad()
            for t in range(1, args.window_length):
                m_next_true = batch_m[:, t]
                phase_curr = batch_phase[:, t - 1] if use_phase else None
                phase_next = batch_phase[:, t] if use_phase else None
                mu, logvar = model.encoder(m_next_true, m_curr, phase_next, phase_curr)
                z = model.reparameterize(mu, logvar)
                m_hat_next = model.decoder(z, m_curr, phase_curr)
                loss, logs = vae_loss(m_hat_next, m_next_true, mu, logvar, 1.0, args.kl_weight)
                (loss / (args.window_length - 1)).backward()
                batch_rec += logs["rec_loss"].item()
                batch_kl += logs["kl_loss"].item()

                # phase is a clock: it ALWAYS advances to the real next value, regardless of
                # whether m_curr was self-fed -- it is never predicted, so there's nothing to
                # substitute or detach here.
                if ss_prob > 0.0:
                    mask = (torch.rand(m_curr.shape[0], device=device) < ss_prob).unsqueeze(-1)
                    m_curr = torch.where(mask, m_hat_next.detach(), m_next_true)
                else:
                    m_curr = m_next_true

            optimizer.step()
            epoch_rec += batch_rec / (args.window_length - 1)
            epoch_kl += batch_kl / (args.window_length - 1)
            num_batches += 1

        scheduler.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"[GMP:{tag}] epoch {epoch:4d}/{args.epochs} "
                f"rec={epoch_rec / num_batches:.6f} kl={epoch_kl / num_batches:.6f} "
                f"ss_prob={ss_prob:.2f} lr={scheduler.get_last_lr()[0]:.2e}"
            )


@torch.no_grad()
def evaluate_teacher_forced(model: MotionCVAE, windows_m: torch.Tensor, windows_phase, device, tag: str):
    """Pure teacher-forced one-step MSE (safeguard #1): compare phase vs no-phase BEFORE looking
    at rollout quality, to separate 'T=100 label is inconsistent' from 'phase concept is bad'."""
    model.eval()
    use_phase = windows_phase is not None
    window_length = windows_m.shape[1]
    total_sqerr, total_n = 0.0, 0
    for start in range(0, windows_m.shape[0], 512):
        batch_m = windows_m[start : start + 512].to(device)
        batch_phase = windows_phase[start : start + 512].to(device) if use_phase else None
        for t in range(1, window_length):
            m_curr, m_next = batch_m[:, t - 1], batch_m[:, t]
            phase_curr = batch_phase[:, t - 1] if use_phase else None
            phase_next = batch_phase[:, t] if use_phase else None
            mu, logvar = model.encoder(m_next, m_curr, phase_next, phase_curr)
            m_hat = model.decoder(mu, m_curr, phase_curr)
            total_sqerr += ((m_hat - m_next) ** 2).sum().item()
            total_n += m_next.numel()
    mse = total_sqerr / total_n
    print(f"[GMP:{tag}] teacher-forced one-step MSE (per-element): {mse:.6f}")
    model.train()
    return mse


@torch.no_grad()
def rollout_and_ablate(
    model: MotionCVAE,
    spec: MotionStateSpec,
    dof_names: list[str],
    seed_state: torch.Tensor,
    period: int,
    device,
    tag: str,
    steps: int = 300,
):
    """Random-z rollout under three phase conditions (safeguard #2):
      (a) real advancing clock -- the actual generation condition
      (b) phase frozen at 0 -- if this looks identical to (a), decoder ignores phase (collapse)
      (c) phase re-randomized every step -- another way to check dependence on phase
    Reports knee-angle autocorrelation rebound for all three, plus a phase-ablation ratio
    analogous to the z-ablation ratio (does using the REAL clock reduce next-step error vs a
    frozen/random one, under teacher forcing conditions -- computed separately below).
    """
    model.eval()
    use_phase = model.phase_dim > 0
    knee_idx = dof_names.index("left_knee_joint") if "left_knee_joint" in dof_names else 0

    def autocorr_rebound(x):
        x = x - x.mean()
        r = np.correlate(x, x, mode="full")
        r = r[len(r) // 2 :]
        r = r / (r[0] + 1e-8)
        return float(r[80:120].max()) if len(r) >= 120 else float("nan")

    def run(phase_mode: str):
        m = seed_state.clone()
        phase_val = 0.0
        knee = np.zeros(steps, dtype=np.float32)
        for t in range(steps):
            knee[t] = spec.split(m)["dof_pos"][0, knee_idx].item()
            z = torch.randn(1, model.latent_dim, device=device)
            if use_phase:
                if phase_mode == "real":
                    phase_val = 2 * math.pi * (t % period) / period
                elif phase_mode == "frozen":
                    phase_val = 0.0
                elif phase_mode == "random":
                    phase_val = float(torch.rand(1).item()) * 2 * math.pi
                phase_curr = torch.tensor([[math.sin(phase_val), math.cos(phase_val)]], device=device)
                m = model.decoder(z, m, phase_curr)
            else:
                m = model.decoder(z, m)
            m = torch.clamp(m, min=-10.0, max=10.0)
        return knee, autocorr_rebound(knee)

    results = {}
    for mode in (["real", "frozen", "random"] if use_phase else ["real"]):
        knee, rebound = run(mode)
        results[mode] = (knee, rebound)
        print(f"[GMP:{tag}] rollout phase_mode={mode}: knee_std={knee.std():.3f} autocorr_rebound={rebound:.3f}")

    if use_phase:
        real_rebound, frozen_rebound = results["real"][1], results["frozen"][1]
        print(
            f"[GMP:{tag}] phase-ablation delta (real - frozen rebound): {real_rebound - frozen_rebound:+.3f} "
            "(near 0 => decoder ignoring phase, same collapse risk as z)"
        )
    model.train()
    return results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    motion_files = []
    for pattern in args.motion_files:
        matches = sorted(glob.glob(pattern))
        motion_files.extend(matches if matches else [pattern])

    window_m_list, window_phase_list = [], []
    motion_dim, num_dofs, dof_names = None, None, None
    for motion_file in motion_files:
        loader = MotionLoader(motion_file=motion_file, device=device)
        m = build_motion_state_sequence(loader)
        motion_dim = m.shape[-1]
        num_dofs = loader.num_dofs
        dof_names = loader.dof_names
        phase = compute_phase_sincos(m.shape[0], args.cycle_period, device)
        window_m_list.append(build_windows(m, args.window_length, args.window_stride))
        window_phase_list.append(build_windows(phase, args.window_length, args.window_stride))
    windows_m = torch.cat(window_m_list, dim=0)
    windows_phase = torch.cat(window_phase_list, dim=0)
    print(f"[GMP] Loaded {len(motion_files)} clip(s), {windows_m.shape[0]} windows, motion_dim={motion_dim}")

    spec = MotionStateSpec(num_dofs=num_dofs)
    seed_state = windows_m[0, 0:1]

    # --- train phase model (phase_dim=2) ---
    model_phase = MotionCVAE(motion_dim, args.latent_dim, args.hidden_sizes, phase_dim=PHASE_DIM).to(device)
    train_one(model_phase, windows_m, windows_phase, args, device, tag="phase")

    # --- train no-phase baseline (phase_dim=0), identical settings otherwise ---
    torch.manual_seed(args.seed)  # same init/data order for as fair a comparison as possible
    model_nophase = MotionCVAE(motion_dim, args.latent_dim, args.hidden_sizes, phase_dim=0).to(device)
    train_one(model_nophase, windows_m, None, args, device, tag="nophase")

    print("\n[GMP] ===== Safeguard 1: teacher-forced MSE, phase vs no-phase =====")
    mse_phase = evaluate_teacher_forced(model_phase, windows_m, windows_phase, device, tag="phase")
    mse_nophase = evaluate_teacher_forced(model_nophase, windows_m, None, device, tag="nophase")
    print(f"[GMP] phase teacher-forced MSE {'BETTER' if mse_phase < mse_nophase else 'WORSE'} than no-phase "
          f"({mse_phase:.6f} vs {mse_nophase:.6f})")

    print("\n[GMP] ===== Safeguard 2: phase ablation + rollout periodicity =====")
    print("[GMP] -- phase model --")
    rollout_and_ablate(model_phase, spec, dof_names, seed_state, args.cycle_period, device, tag="phase")
    print("[GMP] -- no-phase baseline --")
    rollout_and_ablate(model_nophase, spec, dof_names, seed_state, args.cycle_period, device, tag="nophase")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)
    for name, model in [("phase", model_phase), ("nophase", model_nophase)]:
        torch.save(
            {
                "decoder_state_dict": model.decoder.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "motion_dim": motion_dim,
                "latent_dim": args.latent_dim,
                "hidden_sizes": args.hidden_sizes,
                "phase_dim": model.phase_dim,
                "cycle_period": args.cycle_period,
                "motion_files": motion_files,
            },
            f"{args.output_prefix}_{name}.pt",
        )
    print(f"[GMP] Saved {args.output_prefix}_phase.pt and {args.output_prefix}_nophase.pt")


if __name__ == "__main__":
    main()
