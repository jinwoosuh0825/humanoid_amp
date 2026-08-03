# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Same as check_rollout.py, but uses the trained command encoder instead of random z sampling."""

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions"))
from motion_loader import MotionLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvae_model import MotionDecoder, _mlp  # noqa: E402
from motion_state import build_motion_state_sequence  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae_checkpoint", type=str, default="checkpoints/cvae_walk.pt")
    parser.add_argument("--cmd_checkpoint", type=str, default="checkpoints/command_encoder.pt")
    parser.add_argument("--motion_files", type=str, nargs="+", default=["../motions/G1_walk.npz"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--command_vx", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cpu")
    cvae_ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)
    decoder = MotionDecoder(
        motion_dim=cvae_ckpt["motion_dim"], latent_dim=cvae_ckpt["latent_dim"], hidden_sizes=cvae_ckpt["hidden_sizes"]
    )
    decoder.load_state_dict(cvae_ckpt["decoder_state_dict"])
    decoder.eval()

    cmd_ckpt = torch.load(args.cmd_checkpoint, map_location=device, weights_only=False)
    command_encoder = _mlp(cmd_ckpt["obs_dim"], cmd_ckpt["hidden_sizes"], cmd_ckpt["latent_dim"])
    command_encoder.load_state_dict(cmd_ckpt["command_encoder_state_dict"])
    command_encoder.eval()

    motion_files = []
    for pattern in args.motion_files:
        matches = sorted(glob.glob(pattern))
        motion_files.extend(matches if matches else [pattern])

    m_all_list = []
    for f in motion_files:
        loader = MotionLoader(motion_file=f, device=device)
        m_all_list.append(build_motion_state_sequence(loader))
    m_all = torch.cat(m_all_list, dim=0)

    num_frames = m_all.shape[0]
    start_idxs = torch.linspace(0, num_frames - 1, args.num_rollouts).long()
    m = m_all[start_idxs].clone()
    c = torch.tensor([[args.command_vx, 0.0, 0.0]] * args.num_rollouts)

    print(f"[CHECK] command = {c[0].tolist()}, steps={args.steps}, loaded {len(motion_files)} motion clip(s)")
    with torch.no_grad():
        for t in range(args.steps):
            if t % 20 == 0 or t == args.steps - 1:
                base_vel = m[:, 0:3]
                print(f"[CHECK] step {t:3d} | base_lin_vel x per rollout: {[round(v, 3) for v in base_vel[:, 0].tolist()]}")
            obs = torch.cat([c, m], dim=-1)
            z = command_encoder(obs)
            m = decoder(z, m)
            m = torch.clamp(m, min=-10.0, max=10.0)


if __name__ == "__main__":
    main()
