# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
GMP motion-state representation.

Following the GMP paper, a robot reference pose is represented as
    m_t = (v_base_t, w_base_t, q_t, p_key_t, v_key_t)
i.e. base linear velocity, base angular velocity, joint (DOF) positions,
keypoint positions (relative to the base) and keypoint linear velocities.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# same 10 keypoints used by g1_amp_env.py's AMP observation, in the same order
# (this matches the body ordering used when the G1 motion npz files were retargeted)
KEY_BODY_NAMES = [
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "right_hip_yaw_link",
    "left_hip_yaw_link",
    "right_rubber_hand",
    "left_rubber_hand",
    "right_ankle_roll_link",
    "left_ankle_roll_link",
]

REFERENCE_BODY_NAME = "pelvis"

NUM_KEY_BODIES = len(KEY_BODY_NAMES)


@dataclass
class MotionStateSpec:
    """Dimensions of the GMP motion-state vector `m_t` for a given number of DOFs."""

    num_dofs: int
    num_key_bodies: int = NUM_KEY_BODIES

    @property
    def dim(self) -> int:
        # base_lin_vel(3) + base_ang_vel(3) + dof_pos(num_dofs) + key_pos(3*K) + key_vel(3*K)
        return 3 + 3 + self.num_dofs + 3 * self.num_key_bodies + 3 * self.num_key_bodies

    def split(self, m: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split a batched motion-state tensor (..., dim) into its named components."""
        i = 0
        out = {}
        out["base_lin_vel"] = m[..., i : i + 3]
        i += 3
        out["base_ang_vel"] = m[..., i : i + 3]
        i += 3
        out["dof_pos"] = m[..., i : i + self.num_dofs]
        i += self.num_dofs
        out["key_pos"] = m[..., i : i + 3 * self.num_key_bodies].reshape(*m.shape[:-1], self.num_key_bodies, 3)
        i += 3 * self.num_key_bodies
        out["key_vel"] = m[..., i : i + 3 * self.num_key_bodies].reshape(*m.shape[:-1], self.num_key_bodies, 3)
        i += 3 * self.num_key_bodies
        assert i == self.dim
        return out


def compute_motion_state(
    dof_pos: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    base_pos: torch.Tensor,
    key_body_pos: torch.Tensor,
    key_body_lin_vel: torch.Tensor,
) -> torch.Tensor:
    """Build the GMP motion-state vector `m_t` from raw simulation/motion quantities.

    Args:
        dof_pos: (..., num_dofs)
        base_lin_vel: (..., 3) base (pelvis) linear velocity
        base_ang_vel: (..., 3) base (pelvis) angular velocity
        base_pos: (..., 3) base (pelvis) position, used to express keypoints relative to base
        key_body_pos: (..., K, 3) world-frame keypoint positions
        key_body_lin_vel: (..., K, 3) world-frame keypoint linear velocities

    Returns:
        m: (..., dim) motion-state vector
    """
    key_pos_rel = key_body_pos - base_pos.unsqueeze(-2)
    m = torch.cat(
        [
            base_lin_vel,
            base_ang_vel,
            dof_pos,
            key_pos_rel.reshape(*key_pos_rel.shape[:-2], -1),
            key_body_lin_vel.reshape(*key_body_lin_vel.shape[:-2], -1),
        ],
        dim=-1,
    )
    return m


class MotionState:
    """Helper to compute motion-state specs and dimensions for a given MotionLoader."""

    def __init__(self, num_dofs: int):
        self.spec = MotionStateSpec(num_dofs=num_dofs)

    @property
    def dim(self) -> int:
        return self.spec.dim


def build_motion_state_sequence(motion_loader) -> torch.Tensor:
    """Build the full sequence of GMP motion-state vectors `m_0, ..., m_{N-1}` from a MotionLoader.

    Args:
        motion_loader: a `humanoid_amp.motions.MotionLoader` instance (already loaded).

    Returns:
        m: (num_frames, dim) tensor of motion states, on the same device as the loader.
    """
    body_index = motion_loader.get_body_index([REFERENCE_BODY_NAME] + KEY_BODY_NAMES)
    ref_idx, key_idx = body_index[0], body_index[1:]

    base_pos = motion_loader.body_positions[:, ref_idx]
    base_lin_vel = motion_loader.body_linear_velocities[:, ref_idx]
    base_ang_vel = motion_loader.body_angular_velocities[:, ref_idx]
    key_body_pos = motion_loader.body_positions[:, key_idx]
    key_body_lin_vel = motion_loader.body_linear_velocities[:, key_idx]
    dof_pos = motion_loader.dof_positions

    return compute_motion_state(
        dof_pos=dof_pos,
        base_lin_vel=base_lin_vel,
        base_ang_vel=base_ang_vel,
        base_pos=base_pos,
        key_body_pos=key_body_pos,
        key_body_lin_vel=key_body_lin_vel,
    )
