# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Standalone (no Isaac Sim needed) diagnostic: roll the frozen GMP decoder out
auto-regressively for a full episode length and see whether the generated
reference trajectory (in particular base_lin_vel) decays/degenerates over time.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions"))
from motion_loader import MotionLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvae_model import MotionDecoder  # noqa: E402
from motion_state import MotionStateSpec, build_motion_state_sequence  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cvae_walk.pt")
    parser.add_argument("--motion_file", type=str, default="../motions/G1_walk.npz")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    decoder = MotionDecoder(
        motion_dim=checkpoint["motion_dim"], latent_dim=checkpoint["latent_dim"], hidden_sizes=checkpoint["hidden_sizes"]
    )
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    decoder.eval()

    loader = MotionLoader(motion_file=args.motion_file, device=device)
    m_all = build_motion_state_sequence(loader)  # (num_frames, motion_dim)
    spec = MotionStateSpec(num_dofs=loader.num_dofs)

    # seed rollouts from real motion frames at a few different starting times
    num_frames = m_all.shape[0]
    start_idxs = torch.linspace(0, num_frames - 1, args.num_rollouts).long()
    m = m_all[start_idxs].clone()  # (num_rollouts, motion_dim)

    print(f"[CHECK] motion_dim={checkpoint['motion_dim']}, num_rollouts={args.num_rollouts}, steps={args.steps}")
    print(f"[CHECK] ground-truth base_lin_vel norm range in training data: "
          f"{spec.split(m_all)['base_lin_vel'].norm(dim=-1).min():.4f} - "
          f"{spec.split(m_all)['base_lin_vel'].norm(dim=-1).max():.4f} "
          f"(mean {spec.split(m_all)['base_lin_vel'].norm(dim=-1).mean():.4f})")
    print()

    with torch.no_grad():
        for t in range(args.steps):
            if t % 20 == 0 or t == args.steps - 1:
                comp = spec.split(m)
                base_vel_norm = comp["base_lin_vel"].norm(dim=-1)
                dof_pos_norm = comp["dof_pos"].norm(dim=-1)
                print(
                    f"[CHECK] step {t:3d} | base_lin_vel norm (per rollout): "
                    f"{[round(v, 4) for v in base_vel_norm.tolist()]} "
                    f"| mean dof_pos norm: {dof_pos_norm.mean().item():.4f}"
                )
            z = torch.randn(m.shape[0], decoder.latent_dim)
            m = decoder(z, m)


if __name__ == "__main__":
    main()
