# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Generates an auto-regressive CVAE+command-encoder rollout (the same process used online during
RL training) and saves it as a motion npz file (same schema as motions/*.npz), so it can be
inspected with motions/motion_viewer.py completely independently of the RL policy -- i.e. this
checks whether the *reference* the robot is being guided towards is itself natural, decoupled
from how well the RL policy has learned to track it.

Usage:
    python generate_cvae_rollout_npz.py --cvae_checkpoint checkpoints/cvae_walk_v2.pt \
        --cmd_checkpoint checkpoints/command_encoder_v2.pt \
        --motion_files '../motions/lafan1_walk/*.npz' \
        --urdf_path ../usd/g1_29dof_rev_1_0.urdf --mesh_dir ../usd \
        --output ../motions/cvae_rollout_check.npz --steps 300
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
import pinocchio as pin

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions"))
from motion_loader import MotionLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvae_model import MotionDecoder, MotionEncoder, _mlp  # noqa: E402
from motion_state import KEY_BODY_NAMES, MotionStateSpec, build_motion_state_sequence  # noqa: E402

BODY_NAMES = ["pelvis"] + KEY_BODY_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae_checkpoint", type=str, required=True)
    parser.add_argument("--cmd_checkpoint", type=str, default=None)
    parser.add_argument(
        "--random_z",
        action="store_true",
        help="Bypass the command encoder entirely and sample z ~ N(0,1) directly each step "
        "(isolates whether the decoder itself, on its own, produces natural motion).",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Teacher-forced reconstruction: at every step, feed the REAL (m_t, m_{t+1}) pair "
        "into the motion encoder to get z, then decode m_hat_{t+1} from (z, m_t) -- i.e. the "
        "basic CVAE reconstruction task, not an auto-regressive rollout (m_curr is always the "
        "real ground-truth frame, never the model's own previous prediction). Isolates whether "
        "the encoder/decoder pair can reconstruct real motion at all, independent of "
        "auto-regressive drift or z-selection strategy.",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="Frame offset into the first motion file to start from.")
    parser.add_argument("--motion_files", type=str, nargs="+", required=True)
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--mesh_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--command_vx", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=60)
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

    command_encoder = None
    motion_encoder = None
    if args.reconstruct:
        motion_encoder = MotionEncoder(
            motion_dim=cvae_ckpt["motion_dim"], latent_dim=cvae_ckpt["latent_dim"], hidden_sizes=cvae_ckpt["hidden_sizes"]
        )
        motion_encoder.load_state_dict(cvae_ckpt["encoder_state_dict"])
        motion_encoder.eval()
    elif not args.random_z:
        cmd_ckpt = torch.load(args.cmd_checkpoint, map_location=device, weights_only=False)
        command_encoder = _mlp(cmd_ckpt["obs_dim"], cmd_ckpt["hidden_sizes"], cmd_ckpt["latent_dim"])
        command_encoder.load_state_dict(cmd_ckpt["command_encoder_state_dict"])
        command_encoder.eval()

    # seed m_ref from a real motion frame, exactly like the RL env's RSI reset
    motion_files = []
    for pattern in args.motion_files:
        matches = sorted(glob.glob(pattern))
        motion_files.extend(matches if matches else [pattern])
    loader = MotionLoader(motion_file=motion_files[0], device=device)
    m_all = build_motion_state_sequence(loader)
    spec = MotionStateSpec(num_dofs=loader.num_dofs)
    dof_names = loader.dof_names

    command = torch.tensor([[args.command_vx, 0.0, 0.0]])

    dof_positions = np.zeros((args.steps, spec.num_dofs), dtype=np.float32)
    root_lin_vel = np.zeros((args.steps, 3), dtype=np.float32)
    with torch.no_grad():
        if args.reconstruct:
            # teacher-forced: m_curr is always the REAL frame at t, never the model's own
            # previous output -- this measures pure one-step reconstruction quality, not
            # auto-regressive rollout stability.
            end = min(args.start_frame + args.steps + 1, m_all.shape[0])
            for i, t in enumerate(range(args.start_frame, end - 1)):
                m_curr = m_all[t : t + 1]
                m_next_true = m_all[t + 1 : t + 2]
                mu, logvar = motion_encoder(m_next_true, m_curr)
                m_hat_next = decoder(mu, m_curr)  # use the mean z (no sampling noise) for a clean reconstruction
                comp = spec.split(m_hat_next)
                dof_positions[i] = comp["dof_pos"][0].numpy()
                root_lin_vel[i] = comp["base_lin_vel"][0].numpy()
        else:
            m = m_all[0:1].clone()
            for t in range(args.steps):
                comp = spec.split(m)
                dof_positions[t] = comp["dof_pos"][0].numpy()
                root_lin_vel[t] = comp["base_lin_vel"][0].numpy()
                if args.random_z:
                    z = torch.randn(1, decoder.latent_dim)
                else:
                    obs = torch.cat([command, m], dim=-1)
                    z = command_encoder(obs)
                m = decoder(z, m)
                m = torch.clamp(m, min=-10.0, max=10.0)

    # integrate base_lin_vel (world frame, identity root orientation) to get a translating root
    # position, so the rollout actually walks across the scene instead of stepping in place.
    dt = 1.0 / args.fps
    root_pos = np.zeros((args.steps, 3), dtype=np.float32)
    root_pos[0] = [0.0, 0.0, 0.793]  # nominal standing pelvis height
    for t in range(1, args.steps):
        root_pos[t] = root_pos[t - 1] + root_lin_vel[t] * dt
    root_quat_xyzw = np.tile([0.0, 0.0, 0.0, 1.0], (args.steps, 1)).astype(np.float32)  # identity, world-aligned

    # forward kinematics (same approach as motions/convert_lafan1_walk.py)
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

    body_angular_velocities = np.zeros((N, B, 3), dtype=np.float32)  # not needed for viewing; left at zero

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
    print(f"[GMP] Saved {N}-frame CVAE rollout to {args.output}")


if __name__ == "__main__":
    main()
