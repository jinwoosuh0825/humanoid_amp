#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch version of data_convert.py: converts every walk*.csv file from the LAFAN1
Retargeting Dataset (https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
into a separate NPZ file, reusing this repo's own G1 URDF/meshes (usd/g1_29dof_rev_1_0.urdf)
instead of downloading a second copy.

Usage:
    python convert_lafan1_walk.py --csv_dir raw_lafan1/g1 --output_dir . \
        --urdf_path ../usd/g1_29dof_rev_1_0.urdf --mesh_dir ../usd/meshes
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
import pinocchio as pin

DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

BODY_NAMES = [
    "pelvis", "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_elbow_link", "right_elbow_link", "right_hip_yaw_link", "left_hip_yaw_link",
    "right_rubber_hand", "left_rubber_hand", "right_ankle_roll_link", "left_ankle_roll_link",
]


def quaternion_inverse(q):
    w, x, y, z = q
    norm_sq = w * w + x * x + y * y + z * z
    if norm_sq < 1e-8:
        norm_sq = 1e-8
    return np.array([w, -x, -y, -z], dtype=q.dtype) / norm_sq


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=q1.dtype)


def compute_angular_velocity(q_prev, q_next, dt, eps=1e-8):
    # quaternion double-cover: q and -q represent the same rotation, but if adjacent frames
    # happen to be stored in opposite hemispheres, differencing them without aligning first
    # yields a spurious near-pi rotation instead of the true (near-zero) one. Align q_next to
    # q_prev's hemisphere before differencing so we always take the shortest rotation.
    if np.dot(q_prev, q_next) < 0.0:
        q_next = -q_next
    q_inv = quaternion_inverse(q_prev)
    q_rel = quaternion_multiply(q_inv, q_next)
    norm_q_rel = np.linalg.norm(q_rel)
    if norm_q_rel < eps:
        return np.zeros(3, dtype=np.float32)
    q_rel /= norm_q_rel
    w = np.clip(q_rel[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(1.0 - w * w)
    if sin_half < eps:
        return np.zeros(3, dtype=np.float32)
    axis = q_rel[1:] / sin_half
    return (angle / dt) * axis


def convert_one(csv_file: str, urdf_path: str, mesh_dir: str, output_path: str):
    df = pd.read_csv(csv_file, header=None)
    data_orig = df.to_numpy(dtype=np.float32)
    N_orig = data_orig.shape[0]

    root_data_orig = data_orig[:, :7]
    joint_data_orig = data_orig[:, 7:]

    fps_orig = 30
    dt_orig = 1.0 / fps_orig
    t_orig = np.linspace(0, (N_orig - 1) * dt_orig, N_orig)

    fps_new = 60
    dt_new = 1.0 / fps_new
    N_new = 2 * N_orig - 1
    t_new = np.linspace(0, (N_orig - 1) * dt_orig, N_new)

    root_pos_interp = interp1d(t_orig, root_data_orig[:, 0:3], axis=0, kind="linear")(t_new)
    rotations_orig = R.from_quat(root_data_orig[:, 3:7])
    slerp = Slerp(t_orig, rotations_orig)
    root_quat_interp = slerp(t_new).as_quat()
    joint_data = interp1d(t_orig, joint_data_orig, axis=0, kind="linear")(t_new)
    root_data = np.hstack((root_pos_interp, root_quat_interp))

    N, fps, dt = N_new, fps_new, dt_new

    dof_positions = joint_data.copy()
    dof_velocities = np.zeros_like(dof_positions)
    dof_velocities[1:-1] = (dof_positions[2:] - dof_positions[:-2]) / (2 * dt)
    dof_velocities[0] = (dof_positions[1] - dof_positions[0]) / dt
    dof_velocities[-1] = (dof_positions[-1] - dof_positions[-2]) / dt
    dof_velocities = gaussian_filter1d(dof_velocities, sigma=1, axis=0)

    body_names_arr = np.array(BODY_NAMES, dtype=np.str_)
    B = len(BODY_NAMES)
    body_positions = np.zeros((N, B, 3), dtype=np.float32)
    body_rotations = np.zeros((N, B, 4), dtype=np.float32)

    robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir, pin.JointModelFreeFlyer())
    model, data_pk = robot.model, robot.data
    q_pin = pin.neutral(model)

    for i in range(N):
        q_pin[0:3] = root_data[i, 0:3]
        q_pin[3:7] = root_data[i, 3:7]
        q_pin[7 : 7 + joint_data.shape[1]] = joint_data[i, :]
        pin.forwardKinematics(model, data_pk, q_pin)
        pin.updateFramePlacements(model, data_pk)
        for j, link_name in enumerate(BODY_NAMES):
            fid = model.getFrameId(link_name)
            link_tf = data_pk.oMf[fid]
            body_positions[i, j, :] = link_tf.translation
            quat_xyzw = pin.Quaternion(link_tf.rotation)
            body_rotations[i, j, :] = [quat_xyzw.w, quat_xyzw.x, quat_xyzw.y, quat_xyzw.z]

    body_linear_velocities = np.zeros_like(body_positions)
    body_linear_velocities[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2 * dt)
    body_linear_velocities[0] = (body_positions[1] - body_positions[0]) / dt
    body_linear_velocities[-1] = (body_positions[-1] - body_positions[-2]) / dt
    body_linear_velocities = gaussian_filter1d(body_linear_velocities, sigma=1, axis=0)

    body_angular_velocities = np.zeros((N, B, 3), dtype=np.float32)
    for j in range(B):
        quats = body_rotations[:, j, :]
        angular_vels = np.zeros((N, 3), dtype=np.float32)
        angular_vels[0] = compute_angular_velocity(quats[0], quats[1], dt)
        angular_vels[-1] = compute_angular_velocity(quats[-2], quats[-1], dt)
        for k in range(1, N - 1):
            av1 = compute_angular_velocity(quats[k - 1], quats[k], dt)
            av2 = compute_angular_velocity(quats[k], quats[k + 1], dt)
            angular_vels[k] = 0.5 * (av1 + av2)
        body_angular_velocities[:, j, :] = gaussian_filter1d(angular_vels, sigma=1, axis=0)

    np.savez(
        output_path,
        fps=fps,
        dof_names=np.array(DOF_NAMES, dtype=np.str_),
        body_names=body_names_arr,
        dof_positions=dof_positions.astype(np.float32),
        dof_velocities=dof_velocities.astype(np.float32),
        body_positions=body_positions,
        body_rotations=body_rotations,
        body_linear_velocities=body_linear_velocities,
        body_angular_velocities=body_angular_velocities,
    )
    print(f"[convert] {csv_file} ({N_orig} @ 30fps -> {N} @ 60fps) -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--mesh_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="walk*.csv")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, args.pattern)))
    print(f"[convert] Found {len(csv_files)} files matching {args.pattern} in {args.csv_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    for csv_file in csv_files:
        name = os.path.splitext(os.path.basename(csv_file))[0]
        output_path = os.path.join(args.output_dir, f"G1_{name}.npz")
        convert_one(csv_file, args.urdf_path, args.mesh_dir, output_path)


if __name__ == "__main__":
    main()
