# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Offline training script for the GMP motion CVAE.

Trains the motion encoder/decoder on retargeted reference motion clips using reconstruction
(MSE) + KL-divergence loss, following the GMP paper. Only the decoder is needed afterwards
(used frozen, online, during RL training).

v2: trains on multi-step *unrolled windows* (not just independent (m_t, m_t+1) pairs), with a
scheduled-sampling probability that ramps from 0 (full teacher forcing) up to nearly 1.0 (mostly
self-fed) over training and *from the very first epoch*. The original version only substituted a
single step, capped at 30% probability, starting halfway through training -- too weak to teach
the decoder to stay natural over long auto-regressive rollouts (see gmp/check_rollout.py and
gmp/generate_cvae_rollout_npz.py --random_z, which showed the decoder collapsing to a static,
non-cycling pose even with that version). This follows Bengio et al. 2015 ("Scheduled Sampling
for Sequence Prediction using Recurrent Neural Networks", cited by the GMP paper) more faithfully.

Usage:
    python train_cvae.py --motion_files ../motions/lafan1_walk/*.npz --output cvae_walk.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions"))
from motion_loader import MotionLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvae_model import MotionCVAE, vae_loss  # noqa: E402
from motion_state import MotionStateSpec, build_motion_state_sequence  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_files", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr_start", type=float, default=1e-5)
    parser.add_argument("--lr_end", type=float, default=1e-7)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--window_length", type=int, default=20, help="Number of steps unrolled per training window.")
    parser.add_argument("--window_stride", type=int, default=10, help="Stride between consecutive windows.")
    parser.add_argument(
        "--scheduled_sampling_max_prob",
        type=float,
        default=0.95,
        help="Scheduled sampling probability ramps from 0 (epoch 0) to this value (final epoch), linearly.",
    )
    parser.add_argument(
        "--cycle_length",
        type=int,
        default=0,
        help="If > 0, adds an explicit cycle-consistency loss: sample real (m_t, m_{t+cycle_length}) "
        "pairs (cycle_length ~= one real gait cycle, e.g. 100 frames), roll the decoder out "
        "auto-regressively (random z each step, full gradient, no detach) for cycle_length steps "
        "starting from m_t, and penalize the distance between the rollout's final state and the "
        "true m_{t+cycle_length}. Unlike step-wise MSE/KL (which only indirectly hopes a limit "
        "cycle emerges), this directly asks for the one thing we actually want: return to a "
        "similar state after one real cycle. Applied once per epoch (not every batch) since it's "
        "expensive (cycle_length sequential decoder calls with a live graph).",
    )
    parser.add_argument("--cycle_loss_weight", type=float, default=1.0, help="Weight on the cycle-consistency loss.")
    parser.add_argument(
        "--cycle_batch_size", type=int, default=32, help="Number of (m_t, m_t+cycle_length) pairs sampled per epoch."
    )
    parser.add_argument(
        "--detach_self_feed",
        type=lambda s: s.lower() not in ("0", "false", "no"),
        default=True,
        help="If True (default), the self-fed m_curr is detach()'d, so gradient from step t's loss "
        "never propagates back through earlier steps -- only a direct 1-step gradient onto the "
        "shared decoder/encoder weights, regardless of window_length. If False, gradient flows "
        "through the *entire* self-feeding chain (real multi-step BPTT), so loss at late t can "
        "actually shape how earlier steps predict. Set to False (e.g. --detach_self_feed false) "
        "to test whether the lack of long-horizon credit assignment (not vanishing gradient) is "
        "why longer window_length alone didn't help.",
    )
    parser.add_argument(
        "--prior_z_prob",
        type=float,
        default=0.0,
        help="Max probability (ramped from 0 at epoch 0, like scheduled_sampling_max_prob) of "
        "replacing the posterior-sampled z with a pure z ~ N(0,1) prior sample before decoding "
        "(reconstruction loss is still computed against the true next frame). Exposes the decoder "
        "to the same kind of out-of-encoder z it will see at generation time (random_z / "
        "command_encoder), instead of only ever decoding posterior z during training.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_windows(m: torch.Tensor, window_length: int, stride: int) -> torch.Tensor:
    """Slice a (num_frames, motion_dim) sequence into overlapping windows (num_windows, window_length, motion_dim)."""
    num_frames = m.shape[0]
    if num_frames < window_length:
        return torch.empty(0, window_length, m.shape[-1], device=m.device, dtype=m.dtype)
    starts = list(range(0, num_frames - window_length + 1, stride))
    return torch.stack([m[s : s + window_length] for s in starts], dim=0)


@torch.no_grad()
def evaluate_reconstruction(model: MotionCVAE, windows: torch.Tensor, spec: MotionStateSpec, device, batch_size: int = 512):
    """Pure teacher-forced (ss_prob=0) one-step reconstruction error, broken down by
    (a) timestep t within the window, and (b) motion-state component group.

    This isolates the *basic* encoder/decoder reconstruction quality (encoder sees the real
    m_t, m_{t+1} pair every step) from the auto-regressive/scheduled-sampling task, so we can
    tell whether the CVAE's fundamental reconstruction is accurate at all, independent of
    compounding rollout drift.
    """
    model.eval()
    window_length = windows.shape[1]
    group_names = ["base_lin_vel", "base_ang_vel", "dof_pos", "key_pos", "key_vel"]
    per_t_mse = torch.zeros(window_length - 1)
    per_group_sqerr = {g: 0.0 for g in group_names}
    per_group_count = {g: 0 for g in group_names}
    num_windows = windows.shape[0]

    for start in range(0, num_windows, batch_size):
        batch = windows[start : start + batch_size].to(device)
        for t in range(1, window_length):
            m_curr = batch[:, t - 1]
            m_next = batch[:, t]
            m_hat, _, _ = model(m_curr, m_next)
            sqerr = (m_hat - m_next) ** 2  # (B, motion_dim)
            per_t_mse[t - 1] += sqerr.sum().item()

            hat_split = spec.split(m_hat)
            true_split = spec.split(m_next)
            for g in group_names:
                diff2 = (hat_split[g] - true_split[g]) ** 2
                per_group_sqerr[g] += diff2.sum().item()
                per_group_count[g] += diff2.numel()

    total_elems_per_t = num_windows * windows.shape[-1]
    per_t_mse /= total_elems_per_t

    print("[GMP] === Teacher-forced one-step reconstruction diagnostic ===")
    print(f"[GMP] per-timestep MSE (t=1..{window_length - 1}, averaged over all dims & windows):")
    print("  " + " ".join(f"{v:.4f}" for v in per_t_mse.tolist()))
    print("[GMP] per-component-group RMSE (pooled over all t, dims, windows):")
    for g in group_names:
        mse_g = per_group_sqerr[g] / per_group_count[g]
        print(f"  {g:14s} MSE={mse_g:.6f}  RMSE={mse_g ** 0.5:.6f}")
    model.train()


@torch.no_grad()
def rollout_periodicity_check(
    model: MotionCVAE, spec: MotionStateSpec, dof_names: list[str], seed_state: torch.Tensor, device, steps: int = 200
) -> tuple[float, float, float]:
    """Quick random-z (pure prior) auto-regressive rollout using the *current* (mid-training)
    decoder, to catch divergence early and track whether periodicity is emerging over training --
    not just at the very end. Returns (max_abs_state, knee_range, autocorr_rebound)."""
    model.eval()
    knee_idx = dof_names.index("left_knee_joint") if "left_knee_joint" in dof_names else 0
    m = seed_state.clone()
    knee = torch.zeros(steps)
    max_abs = 0.0
    for t in range(steps):
        comp = spec.split(m)
        knee[t] = comp["dof_pos"][0, knee_idx]
        z = torch.randn(1, model.latent_dim, device=device)
        m = model.decoder(z, m)
        m = torch.clamp(m, min=-10.0, max=10.0)
        max_abs = max(max_abs, m.abs().max().item())
    knee_np = knee.numpy()
    x = knee_np - knee_np.mean()
    r = np.correlate(x, x, mode="full")
    r = r[len(r) // 2 :]
    r = r / (r[0] + 1e-8)
    rebound = float(r[80:120].max()) if steps >= 120 else float("nan")
    model.train()
    return max_abs, float(knee_np.max() - knee_np.min()), rebound


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # expand any glob patterns (e.g. lafan1_walk/*.npz) and load every clip separately --
    # windows are built *within* each clip only, so no window ever spans a clip boundary.
    motion_files = []
    for pattern in args.motion_files:
        matches = sorted(glob.glob(pattern))
        motion_files.extend(matches if matches else [pattern])

    window_list = []
    full_sequences = []  # kept separately (undivided) so we can sample (m_t, m_{t+cycle_length}) pairs
    motion_dim = None
    num_dofs = None
    dof_names = None
    total_frames = 0
    for motion_file in motion_files:
        loader = MotionLoader(motion_file=motion_file, device=device)
        m = build_motion_state_sequence(loader)  # (num_frames, motion_dim)
        motion_dim = m.shape[-1]
        num_dofs = loader.num_dofs
        dof_names = loader.dof_names
        total_frames += m.shape[0]
        full_sequences.append(m)
        window_list.append(build_windows(m, args.window_length, args.window_stride))
    windows = torch.cat(window_list, dim=0)  # (num_windows, window_length, motion_dim)
    num_windows = windows.shape[0]

    # pool of valid (sequence_idx, start_idx) pairs for the cycle-consistency loss
    cycle_pairs = []
    if args.cycle_length > 0:
        for seq_idx, seq in enumerate(full_sequences):
            for start_idx in range(0, seq.shape[0] - args.cycle_length):
                cycle_pairs.append((seq_idx, start_idx))
        print(f"[GMP] cycle-consistency loss enabled: cycle_length={args.cycle_length}, {len(cycle_pairs)} candidate pairs")
    print(
        f"[GMP] Loaded {len(motion_files)} motion clip(s), {total_frames} total frames, "
        f"{num_windows} windows of length {args.window_length} (stride {args.window_stride}), "
        f"motion_dim={motion_dim}"
    )

    spec = MotionStateSpec(num_dofs=num_dofs)
    rollout_seed_state = windows[0, 0:1]  # a real first frame, reused every periodicity check

    model = MotionCVAE(motion_dim=motion_dim, latent_dim=args.latent_dim, hidden_sizes=args.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start)
    gamma = (args.lr_end / args.lr_start) ** (1.0 / max(1, args.epochs - 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(args.epochs):
        perm = torch.randperm(num_windows, device=device)
        # scheduled sampling probability: ramps up linearly from epoch 0 (unlike the old version,
        # which stayed at teacher-forcing for the first half of training and capped at 0.3).
        ss_prob = args.scheduled_sampling_max_prob * (epoch / max(1, args.epochs - 1))
        # prior_z_prob ramps up the same way -- pushing full prior-z substitution from epoch 0
        # (before the decoder has learned anything from m_curr yet) risks destabilizing early
        # training, so ramp it in gradually just like ss_prob.
        prior_z_prob = args.prior_z_prob * (epoch / max(1, args.epochs - 1))

        epoch_rec, epoch_kl, num_batches = 0.0, 0.0, 0
        for start in range(0, num_windows, args.batch_size):
            idx = perm[start : start + args.batch_size]
            batch = windows[idx]  # (B, window_length, motion_dim)

            m_curr = batch[:, 0]  # always the real first frame of the window
            batch_rec, batch_kl = 0.0, 0.0
            optimizer.zero_grad()
            total_loss = 0.0
            for t in range(1, args.window_length):
                m_next_true = batch[:, t]
                mu, logvar = model.encoder(m_next_true, m_curr)
                z = model.reparameterize(mu, logvar)
                if prior_z_prob > 0.0:
                    prior_mask = (torch.rand(z.shape[0], device=device) < prior_z_prob).unsqueeze(-1)
                    z = torch.where(prior_mask, torch.randn_like(z), z)
                m_hat_next = model.decoder(z, m_curr)
                loss, logs = vae_loss(m_hat_next, m_next_true, mu, logvar, args.rec_weight, args.kl_weight)
                # accumulate (don't backward yet): if detach_self_feed is False, gradient must be
                # able to flow through the *entire* self-feeding chain, which requires keeping the
                # whole window's graph alive until one single backward() call at the end. Calling
                # backward() per-step (the old behavior) would free each step's graph immediately,
                # silently breaking multi-step credit assignment even if detach were removed.
                total_loss = total_loss + loss
                batch_rec += logs["rec_loss"].item()
                batch_kl += logs["kl_loss"].item()

                # scheduled sampling: feed the model's own prediction back in as the next step's
                # conditioning frame for a random subset of the batch, instead of always using the
                # real ground-truth next frame. Whether this prediction is detached (cutting the
                # gradient chain at this point, so loss at later t can never shape earlier-t
                # behavior) or left attached (real multi-step BPTT) is controlled by
                # --detach_self_feed.
                m_hat_next_fed = m_hat_next.detach() if args.detach_self_feed else m_hat_next
                if ss_prob > 0.0:
                    mask = (torch.rand(m_curr.shape[0], device=device) < ss_prob).unsqueeze(-1)
                    m_curr = torch.where(mask, m_hat_next_fed, m_next_true)
                else:
                    m_curr = m_next_true

            # normalize so the total gradient magnitude doesn't scale with window_length
            (total_loss / (args.window_length - 1)).backward()
            optimizer.step()
            epoch_rec += batch_rec / (args.window_length - 1)
            epoch_kl += batch_kl / (args.window_length - 1)
            num_batches += 1

        cycle_loss_val = None
        if args.cycle_length > 0:
            # explicit cycle-consistency step: sample real (m_t, m_{t+cycle_length}) pairs and
            # directly penalize the auto-regressive rollout's distance from the true future state
            # one real cycle later -- unlike step-wise MSE, this asks for the specific thing we
            # want (return to a similar state after one cycle), not just "don't be wrong locally".
            chosen = [cycle_pairs[i] for i in torch.randint(0, len(cycle_pairs), (args.cycle_batch_size,)).tolist()]
            m_t = torch.stack([full_sequences[si][st] for si, st in chosen], dim=0).to(device)
            m_target = torch.stack([full_sequences[si][st + args.cycle_length] for si, st in chosen], dim=0).to(device)
            optimizer.zero_grad()
            m_pred = m_t
            # sample z ONCE per rollout (not every step): resampling every step would make m_pred
            # a stochastic accumulation of cycle_length independent random draws compared against
            # a single fixed target, giving the loss an irreducible noise floor unrelated to
            # whether the decoder actually learns a return-to-cycle path -- the only way to reduce
            # that floor would be for the decoder to learn to ignore z entirely, the opposite of
            # what we want.
            z_fixed = torch.randn(m_pred.shape[0], args.latent_dim, device=device)
            for _ in range(args.cycle_length):
                m_pred = model.decoder(z_fixed, m_pred)
            cycle_loss = torch.nn.functional.mse_loss(m_pred, m_target)
            (args.cycle_loss_weight * cycle_loss).backward()
            optimizer.step()
            cycle_loss_val = cycle_loss.item()

        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            cycle_str = f" cycle_loss={cycle_loss_val:.4f}" if cycle_loss_val is not None else ""
            print(
                f"[GMP] epoch {epoch:4d}/{args.epochs} "
                f"rec={epoch_rec / num_batches:.6f} kl={epoch_kl / num_batches:.6f} "
                f"ss_prob={ss_prob:.2f} prior_z_prob={prior_z_prob:.2f} lr={scheduler.get_last_lr()[0]:.2e}{cycle_str}"
            )
        if epoch % 40 == 0 or epoch == args.epochs - 1:
            max_abs, knee_range, rebound = rollout_periodicity_check(model, spec, dof_names, rollout_seed_state, device)
            print(
                f"[GMP]   rollout-check @ epoch {epoch}: max_abs_state={max_abs:.2f} "
                f"knee_range={knee_range:.3f} autocorr_rebound={rebound:.3f}"
            )

    evaluate_reconstruction(model, windows, spec, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "decoder_state_dict": model.decoder.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "motion_dim": motion_dim,
            "latent_dim": args.latent_dim,
            "hidden_sizes": args.hidden_sizes,
            "motion_files": motion_files,
        },
        args.output,
    )
    print(f"[GMP] Saved frozen decoder checkpoint to {args.output}")


if __name__ == "__main__":
    main()
