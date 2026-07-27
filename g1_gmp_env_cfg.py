# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 locomotion environment guided by a frozen CVAE Generative Motion Prior (GMP),
following "Natural Humanoid Robot Locomotion with Generative Motion Prior" (Zhang et al., IROS 2025).

Unlike g1_amp_env_cfg.py (discriminator/style-reward based AMP), this environment has no
discriminator: the reward is computed directly from a frozen, pretrained CVAE motion decoder
that autoregressively generates a reference trajectory online, plus standard regularization
terms. It is meant to be trained with plain PPO (see agents/skrl_g1_walk_gmp_cfg.yaml).
"""

from __future__ import annotations

import os
from dataclasses import MISSING
from .g1_cfg import G1_CFG


from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")
GMP_CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gmp", "checkpoints")


@configclass
class G1GmpEnvCfg(DirectRLEnvCfg):
    """G1 locomotion environment config guided by a frozen CVAE motion prior (base class)."""

    # regularization reward (same terms as g1_amp_env_cfg.py, filled in with non-zero weights
    # since there is no discriminator here to provide a style reward)
    rew_termination = -200.0
    rew_action_l2 = -0.01
    rew_joint_pos_limits = -10.0
    rew_joint_acc_l2 = -2.5e-7
    rew_joint_vel_l2 = -1.0e-3

    # motion guidance reward (GMP paper Eq. 11-13): r_guidance = r_dof + r_keypos,
    # r_dof = exp(-guidance_temp * ||q - q_ref||), r_keypos = exp(-guidance_temp * ||p - p_ref||)
    rew_dof_guidance = 0.5
    rew_keypos_guidance = 0.5
    guidance_temp = 0.7

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces (same observation as AMP env; no amp_observation_space needed, no discriminator)
    observation_space = 71 + 3 * 10
    action_space = 29
    state_space = 0

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    gmp_checkpoint: str = MISSING
    """Path to a frozen CVAE decoder checkpoint produced by gmp/train_cvae.py."""

    reference_body = "pelvis"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class G1GmpWalkEnvCfg(G1GmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "G1_walk.npz")
    gmp_checkpoint = os.path.join(GMP_CHECKPOINTS_DIR, "cvae_walk.pt")
