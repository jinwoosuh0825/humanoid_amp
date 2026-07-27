# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 locomotion environment guided by a frozen CVAE Generative Motion Prior (GMP), replacing the
discriminator/style-reward of AMP with a motion-guidance reward computed against an online,
auto-regressively generated reference trajectory (see gmp/cvae_model.py, gmp/motion_state.py).
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .g1_amp_env import compute_obs
from .g1_gmp_env_cfg import G1GmpEnvCfg
from .motions import MotionLoader
from .gmp.cvae_model import MotionDecoder
from .gmp.motion_state import KEY_BODY_NAMES, REFERENCE_BODY_NAME, MotionStateSpec, compute_motion_state


class G1GmpEnv(DirectRLEnv):
    cfg: G1GmpEnvCfg

    def __init__(self, cfg: G1GmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # load reference motion (used for reset state sampling / RSI, and to seed the GMP reference)
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        key_body_names = KEY_BODY_NAMES
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([REFERENCE_BODY_NAME])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # load frozen GMP motion decoder
        checkpoint = torch.load(self.cfg.gmp_checkpoint, map_location=self.device, weights_only=False)
        self.motion_state_spec = MotionStateSpec(num_dofs=self._motion_loader.num_dofs)
        assert self.motion_state_spec.dim == checkpoint["motion_dim"], (
            f"GMP checkpoint motion_dim ({checkpoint['motion_dim']}) does not match the motion file's "
            f"derived motion-state dim ({self.motion_state_spec.dim}); was it trained on a different motion file?"
        )
        self.gmp_decoder = MotionDecoder(
            motion_dim=checkpoint["motion_dim"],
            latent_dim=checkpoint["latent_dim"],
            hidden_sizes=checkpoint["hidden_sizes"],
        ).to(self.device)
        self.gmp_decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.gmp_decoder.eval()
        for p in self.gmp_decoder.parameters():
            p.requires_grad_(False)
        print(
            f"[GMP] Loaded frozen decoder from {self.cfg.gmp_checkpoint} "
            f"(motion_dim={checkpoint['motion_dim']}, latent_dim={checkpoint['latent_dim']})"
        )

        # per-env online GMP reference motion state (auto-regressively generated, open-loop)
        self.m_ref = torch.zeros((self.num_envs, checkpoint["motion_dim"]), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # advance the frozen GMP reference trajectory by one step: auto-regressive, open-loop
        # (independent of the policy's actual state), following Fig. 2(c) of the GMP paper.
        with torch.no_grad():
            z = torch.randn(self.num_envs, self.gmp_decoder.latent_dim, device=self.device)
            self.m_ref = self.gmp_decoder(z, self.m_ref)

        ref = self.motion_state_spec.split(self.m_ref)
        # reference dof positions are in the motion file's (npz) DOF order; reorder to match
        # the simulation's joint order (same mapping used for RSI resets).
        q_ref = ref["dof_pos"][:, self.motion_dof_indexes]
        p_ref = ref["key_pos"]  # already relative to base, same key-body order as self.key_body_indexes

        q = self.robot.data.joint_pos
        base_pos = self.robot.data.body_pos_w[:, self.ref_body_index]
        p = self.robot.data.body_pos_w[:, self.key_body_indexes] - base_pos.unsqueeze(1)

        total_reward, reward_log = compute_rewards(
            self.cfg.rew_termination,
            self.cfg.rew_action_l2,
            self.cfg.rew_joint_pos_limits,
            self.cfg.rew_joint_acc_l2,
            self.cfg.rew_joint_vel_l2,
            self.cfg.rew_dof_guidance,
            self.cfg.rew_keypos_guidance,
            self.cfg.guidance_temp,
            self.reset_terminated,
            self.actions,
            q,
            self.robot.data.soft_joint_pos_limits,
            self.robot.data.joint_acc,
            self.robot.data.joint_vel,
            q_ref,
            p,
            p_ref,
        )
        self.extras["log"] = reward_log
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self._seed_motion_reference(env_ids, start=True)
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.05  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, self.motion_ref_body_index]
        root_state[:, 7:10] = body_linear_velocities[:, self.motion_ref_body_index]
        root_state[:, 10:13] = body_angular_velocities[:, self.motion_ref_body_index]

        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # seed the GMP online reference state m_ref from this same sampled mocap frame (m_0),
        # in the motion file's own DOF order (matching how the CVAE was trained).
        m0 = compute_motion_state(
            dof_pos=dof_positions,
            base_lin_vel=body_linear_velocities[:, self.motion_ref_body_index],
            base_ang_vel=body_angular_velocities[:, self.motion_ref_body_index],
            base_pos=body_positions[:, self.motion_ref_body_index],
            key_body_pos=body_positions[:, self.motion_key_body_indexes],
            key_body_lin_vel=body_linear_velocities[:, self.motion_key_body_indexes],
        )
        self.m_ref[env_ids] = m0

        return root_state, dof_pos, dof_vel

    def _seed_motion_reference(self, env_ids: torch.Tensor, start: bool = True):
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        (
            dof_positions,
            _,
            body_positions,
            _,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        m0 = compute_motion_state(
            dof_pos=dof_positions,
            base_lin_vel=body_linear_velocities[:, self.motion_ref_body_index],
            base_ang_vel=body_angular_velocities[:, self.motion_ref_body_index],
            base_pos=body_positions[:, self.motion_ref_body_index],
            key_body_pos=body_positions[:, self.motion_key_body_indexes],
            key_body_lin_vel=body_linear_velocities[:, self.motion_key_body_indexes],
        )
        self.m_ref[env_ids] = m0


@torch.jit.script
def compute_rewards(
    rew_scale_termination: float,
    rew_scale_action_l2: float,
    rew_scale_joint_pos_limits: float,
    rew_scale_joint_acc_l2: float,
    rew_scale_joint_vel_l2: float,
    rew_scale_dof_guidance: float,
    rew_scale_keypos_guidance: float,
    guidance_temp: float,
    reset_terminated: torch.Tensor,
    actions: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    joint_acc: torch.Tensor,
    joint_vel: torch.Tensor,
    q_ref: torch.Tensor,
    p: torch.Tensor,
    p_ref: torch.Tensor,
):
    rew_termination = rew_scale_termination * reset_terminated.float()
    rew_action_l2 = rew_scale_action_l2 * torch.sum(torch.square(actions), dim=1)

    out_of_limits = -(joint_pos - soft_joint_pos_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (joint_pos - soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
    rew_joint_pos_limits = rew_scale_joint_pos_limits * torch.sum(out_of_limits, dim=1)

    rew_joint_acc_l2 = rew_scale_joint_acc_l2 * torch.sum(torch.square(joint_acc), dim=1)
    rew_joint_vel_l2 = rew_scale_joint_vel_l2 * torch.sum(torch.square(joint_vel), dim=1)

    # GMP motion guidance reward (Eq. 11-13 of the GMP paper)
    dof_err = torch.norm(joint_pos - q_ref, dim=-1)
    rew_dof_guidance = rew_scale_dof_guidance * torch.exp(-guidance_temp * dof_err)

    keypos_err = torch.norm((p - p_ref).reshape(p.shape[0], -1), dim=-1)
    rew_keypos_guidance = rew_scale_keypos_guidance * torch.exp(-guidance_temp * keypos_err)

    total_reward = (
        rew_termination
        + rew_action_l2
        + rew_joint_pos_limits
        + rew_joint_acc_l2
        + rew_joint_vel_l2
        + rew_dof_guidance
        + rew_keypos_guidance
    )

    log = {
        "rew_termination": (rew_termination).mean(),
        "rew_action_l2": (rew_action_l2).mean(),
        "rew_joint_pos_limits": (rew_joint_pos_limits).mean(),
        "rew_joint_acc_l2": (rew_joint_acc_l2).mean(),
        "rew_joint_vel_l2": (rew_joint_vel_l2).mean(),
        "rew_dof_guidance": (rew_dof_guidance).mean(),
        "rew_keypos_guidance": (rew_keypos_guidance).mean(),
    }
    return total_reward, log
