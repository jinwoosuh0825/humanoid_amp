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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")
GMP_CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gmp", "checkpoints")


@configclass
class G1GmpEnvCfg(DirectRLEnvCfg):
    """G1 locomotion environment config guided by a frozen CVAE motion prior (base class)."""

    # regularization reward (same terms as g1_amp_env_cfg.py, filled in with non-zero weights
    # since there is no discriminator here to provide a style reward).
    #
    # NOTE: r_reg is not GMP's own contribution (the GMP paper studies motion *guidance*, and
    # its regularization values were tuned for a different robot, NAVIAI: 41 DoF, 60kg). For
    # these generic stability terms we instead follow Unitree's own real, hardware-validated G1
    # PPO config (unitree_rl_gym/legged_gym/envs/g1/g1_config.py + base legged_robot_config.py),
    # since it's tuned for this exact robot.
    rew_termination = -200.0
    rew_joint_pos_limits = -5.0  # Unitree G1: dof_pos_limits
    rew_joint_acc_l2 = -2.5e-7  # Unitree G1: dof_acc
    rew_joint_vel_l2 = -1.0e-3  # Unitree G1: dof_vel

    # disabled -- replaced by rew_action_rate below (Unitree G1 doesn't penalize raw action
    # magnitude, only the rate of change between consecutive actions)
    rew_action_l2 = 0.0
    rew_action_rate = -0.01  # Unitree G1: action_rate, computed on (actions - prev_actions)

    # Unitree G1: lin_vel_z / ang_vel_xy, penalize the base moving vertically or rocking
    # roll/pitch, in the base's own body frame (not in the GMP paper).
    rew_lin_vel_z = -2.0
    rew_ang_vel_xy = -0.05

    # Unitree G1: only_positive_rewards -- clip the total reward at 0 before it reaches PPO.
    # Prevents large negative sums (e.g. from termination) from dominating the learning signal.
    only_positive_rewards = True

    # GMP paper Table I: "Projected Gravity", weight -6.0. Penalizes tilting away from upright;
    # also prevents the handstand exploit (height-only termination satisfied while flipped over).
    rew_projected_gravity = -6.0

    # motion guidance reward (GMP paper Eq. 11-13): r_guidance = r_dof + r_keypos (no separate
    # weighting in the paper -- i.e. both terms implicitly have weight 1.0),
    # r_dof = exp(-guidance_temp * ||q - q_ref||), r_keypos = exp(-guidance_temp * ||p - p_ref||)
    rew_dof_guidance = 1.0
    rew_keypos_guidance = 1.0
    guidance_temp = 0.7

    # NOTE: previously had a `rew_base_vel_guidance` term here rewarding the robot for matching
    # m_ref's own base_lin_vel (a stand-in task reward, back when there was no fixed command).
    # Removed now that rew_task_lin_vel (below) exists: keeping both would pull the robot toward
    # two different targets at once (m_ref's often-lower achieved speed vs. the fixed command).

    # GMP paper Table I: "Task Reward" -- Linear Velocity (weight 3.0), Angular Velocity
    # (weight 2.5): exp(-4*||v - c||^2). Previously skipped since we had no command; now that
    # a fixed command (vx, vy, wz) already exists (for the command encoder), this rewards the
    # *robot itself* for matching that same fixed target directly, instead of only indirectly
    # via m_ref (rew_base_vel_guidance above, whose own tracking of the command is imperfect).
    rew_task_lin_vel = 3.0
    rew_task_ang_vel = 2.5
    task_tracking_temp = 4.0
    command_wz = 0.0  # target yaw rate (0 = walk straight)

    # GMP paper Table I: "Feet Air Time" (weight 20) and "No Fly" (weight 0.8). These are the
    # only reward terms that directly require an alternating stepping gait (lifting feet off
    # the ground) rather than e.g. sliding/shuffling while matching base velocity.
    rew_feet_air_time = 20.0
    rew_no_fly = 0.8
    feet_air_time_threshold = 0.5
    feet_body_regex = ".*_ankle_roll_link"

    # NOTE: not in the GMP paper or Unitree's config -- rew_feet_air_time/rew_no_fly above sum
    # over both feet, so a policy can satisfy them by stepping normally with one leg while
    # dragging/planting the other. This directly penalizes an imbalance in each foot's per-episode
    # contact-event count, to discourage that one-legged-gait exploit.
    rew_foot_balance = -2.0

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

    command_encoder_checkpoint: str = MISSING
    """Path to a frozen command encoder checkpoint produced by gmp/train_command_encoder.py."""

    # fixed target command (vx, vy) for the command encoder -- see NOTE in train_command_encoder.py:
    # we don't need arbitrary user-controlled commands, just a stable non-collapsing target speed.
    command_vx = 0.5
    command_vy = 0.0

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

    # contact sensor (feet air time / no-fly rewards)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True
    )


@configclass
class G1GmpWalkEnvCfg(G1GmpEnvCfg):
    # RSI reset sampling still uses the single original clip (unrelated to the CVAE/command
    # encoder's own training data below -- this is just where a robot's pose gets reset to).
    motion_file = os.path.join(MOTIONS_DIR, "G1_walk.npz")
    # CVAE decoder + command encoder now trained on the expanded 12-clip LAFAN1 walk dataset
    # (motions/lafan1_walk/*.npz, ~173,730 frames) instead of the original single 399-frame clip.
    gmp_checkpoint = os.path.join(GMP_CHECKPOINTS_DIR, "cvae_walk_v2.pt")
    command_encoder_checkpoint = os.path.join(GMP_CHECKPOINTS_DIR, "command_encoder_v2.pt")
