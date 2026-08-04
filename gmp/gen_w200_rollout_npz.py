# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Generates the window_length=200 self-feed command_encoder's auto-regressive rollout and saves it
as a motion npz (same schema as motions/*.npz) for GIF rendering via render_motion_video.py.
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
import pinocchio as pin

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions"))
from motion_loader import MotionLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvae_model import MotionDecoder, _mlp  # noqa: E402
from motion_state import KEY_BODY_NAMES, MotionStateSpec, build_motion_state_sequence  # noqa: E402

BODY_NAMES = ["pelvis"] + KEY_BODY_NAMES


class GaussianCommandEncoder(nn.Module):
    def __init__(self, motion_dim, latent_dim, hidden_sizes, min_std):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_logvar = 2 * math.log(min_std)
        self.net = _mlp(motion_dim, hidden_sizes, 2 * latent_dim)

    def forward(self, m):
        out = self.net(m)
        mean, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, self.min_logvar, 2.0)
        return mean, logvar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae_checkpoint", type=str, required=True)
    parser.add_argument("--cmd_checkpoint", type=str, required=True)
    parser.add_argument("--motion_files", type=str, nargs="+", required=True)
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--mesh_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--start_frame", type=int, default=3300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    cvae_ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)
    decoder = MotionDecoder(
        motion_dim=cvae_ckpt["motion_dim"], latent_dim=cvae_ckpt["latent_dim"], hidden_sizes=cvae_ckpt["hidden_sizes"]
    )
    decoder.load_state_dict(cvae_ckpt["decoder_state_dict"])
    decoder.eval()

    cmd_ckpt = torch.load(args.cmd_checkpoint, map_location=device, weights_only=False)
    command_encoder = GaussianCommandEncoder(
        cmd_ckpt["motion_dim"], cmd_ckpt["latent_dim"], cmd_ckpt["hidden_sizes"], cmd_ckpt["min_std"]
    )
    command_encoder.load_state_dict(cmd_ckpt["command_encoder_state_dict"])
    command_encoder.eval()

    motion_files = []
    for pattern in args.motion_files:
        matches = sorted(glob.glob(pattern))
        motion_files.extend(matches if matches else [pattern])
    loader = MotionLoader(motion_file=motion_files[0], device=device)
    m_all = build_motion_state_sequence(loader)
    spec = MotionStateSpec(num_dofs=loader.num_dofs)
    dof_names = loader.dof_names

    dof_positions = np.zeros((args.steps, spec.num_dofs), dtype=np.float32)
    root_lin_vel = np.zeros((args.steps, 3), dtype=np.float32)
    with torch.no_grad():
        m = m_all[args.start_frame : args.start_frame + 1].clone()
        for t in range(args.steps):
            comp = spec.split(m)
            dof_positions[t] = comp["dof_pos"][0].numpy()
            root_lin_vel[t] = comp["base_lin_vel"][0].numpy()
            mean, logvar = command_encoder(m)
            std = (0.5 * logvar).exp()
            z = mean + std * torch.randn_like(mean)
            m = decoder(z, m)
            m = torch.clamp(m, min=-10.0, max=10.0)

    dt = 1.0 / args.fps
    root_pos = np.zeros((args.steps, 3), dtype=np.float32)
    root_pos[0] = [0.0, 0.0, 0.793]
    for t in range(1, args.steps):
        root_pos[t] = root_pos[t - 1] + root_lin_vel[t] * dt
    root_quat_xyzw = np.tile([0.0, 0.0, 0.0, 1.0], (args.steps, 1)).astype(np.float32)

    robot = pin.RobotWrapper.BuildFromURDF(args.urdf_path, args.mesh_dir, pin.JointModelFreeFlyer())
    model, data_pk = robot.model, robot.data
    q_pin = pin.neutral(model)

    N = args.steps
    B = len(BODY_NAMES)
    body_positions = np.zeros((N, B, 3), dtype=np.float32)
    body_rotations = np.zeros((N, B, 4), dtype=np.float32)

    for i in range(N):
        q_pin[0:3] = root_pos[i]
        q_pin[3:7] = root_quat_xyzw[i]
        q_pin[7 : 7 + dof_positions.shape[1]] = dof_positions[i]
        pin.forwardKinematics(model, data_pk, q_pin)
        pin.updateFramePlacements(model, data_pk)
        for j, link_name in enumerate(BODY_NAMES):
            fid = model.getFrameId(link_name)
            link_tf = data_pk.oMf[fid]
            body_positions[i, j, :] = link_tf.translation
            quat_xyzw = pin.Quaternion(link_tf.rotation)
            body_rotations[i, j, :] = [quat_xyzw.w, quat_xyzw.x, quat_xyzw.y, quat_xyzw.z]

    dof_velocities = np.zeros_like(dof_positions)
    dof_velocities[1:-1] = (dof_positions[2:] - dof_positions[:-2]) / (2 * dt)
    dof_velocities[0] = (dof_positions[1] - dof_positions[0]) / dt
    dof_velocities[-1] = (dof_positions[-1] - dof_positions[-2]) / dt
    dof_velocities = gaussian_filter1d(dof_velocities, sigma=1, axis=0)

    body_linear_velocities = np.zeros_like(body_positions)
    body_linear_velocities[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2 * dt)
    body_linear_velocities[0] = (body_positions[1] - body_positions[0]) / dt
    body_linear_velocities[-1] = (body_positions[-1] - body_positions[-2]) / dt
    body_linear_velocities = gaussian_filter1d(body_linear_velocities, sigma=1, axis=0)

    body_angular_velocities = np.zeros((N, B, 3), dtype=np.float32)

    np.savez(
        args.output,
        fps=args.fps,
        dof_names=np.array(dof_names, dtype=np.str_),
        body_names=np.array(BODY_NAMES, dtype=np.str_),
        dof_positions=dof_positions,
        dof_velocities=dof_velocities,
        body_positions=body_positions,
        body_rotations=body_rotations,
        body_linear_velocities=body_linear_velocities,
        body_angular_velocities=body_angular_velocities,
    )
    print(f"[GMP] Saved {N}-frame w200 self-feed rollout to {args.output}")


if __name__ == "__main__":
    main()
