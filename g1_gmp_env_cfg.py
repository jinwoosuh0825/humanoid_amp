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
    # r_reg partial ablation (2026-08-05): the paper's 3-term breakdown is r_guidance + r_task +
    # r_reg. First tried r_reg fully disabled -- comparing to the previous GMP run (reg on, old
    # broken reference generator), that run stopped falling by ~10k steps, while this reg-off run
    # was still falling immediately at 40k steps and rew_termination had completely flatlined by
    # 15k. That's strong evidence the "immediate stability" subset of r_reg (action smoothness,
    # base bounce/rocking) is actually load-bearing for learning not to fall quickly, not just a
    # sim-to-real nicety -- so those 3 are back on. The other 3 (joint_pos_limits, joint_acc_l2,
    # joint_vel_l2 -- more about long-run joint wear than immediate balance) stay off.
    rew_termination = -200.0
    rew_joint_pos_limits = 0.0  # was -5.0 (Unitree G1: dof_pos_limits) -- still off
    rew_joint_acc_l2 = 0.0  # was -2.5e-7 (Unitree G1: dof_acc) -- still off
    rew_joint_vel_l2 = 0.0  # was -1.0e-3 (Unitree G1: dof_vel) -- still off

    # disabled -- replaced by rew_action_rate below (Unitree G1 doesn't penalize raw action
    # magnitude, only the rate of change between consecutive actions)
    rew_action_l2 = 0.0
    rew_action_rate = -0.01  # re-enabled (Unitree G1: action_rate, on actions - prev_actions)

    # Unitree G1: lin_vel_z / ang_vel_xy, penalize the base moving vertically or rocking
    # roll/pitch, in the base's own body frame (not in the GMP paper).
    rew_lin_vel_z = -2.0  # re-enabled
    rew_ang_vel_xy = -0.05  # re-enabled

    # Unitree G1: only_positive_rewards -- clip the total reward at 0 before it reaches PPO.
    # Prevents large negative sums (e.g. from termination) from dominating the learning signal.
    only_positive_rewards = True

    # GMP paper Table I: "Projected Gravity", weight -6.0. Penalizes tilting away from upright;
    # also prevents the handstand exploit (height-only termination satisfied while flipped over).
    rew_projected_gravity = -6.0

    # motion guidance reward (GMP paper Eq. 11-13): r_guidance = r_dof + r_keypos (no separate
    # weighting in the paper -- i.e. both terms implicitly have weight 1.0),
    # r_dof = exp(-guidance_temp * ||q - q_ref||), r_keypos = exp(-guidance_temp * ||p - p_ref||)
    #
    # Raised from the paper's 1.0/1.0 (2026-08-05): with paper weights, guidance's max possible
    # contribution (2.0) was already less than half of task's (5.5), and once training got going
    # task_lin_vel sat near its ceiling while dof_guidance flatlined near its floor -- policy
    # ended up ignoring the reference and finding a locally-efficient but asymmetric gait (one
    # leg doing most of the work) that still satisfies task/stability rewards. Since the
    # reference itself already walks with a natural, symmetric, periodic gait (verified
    # separately), the fix is to make tracking it actually matter, not to add a new penalty for
    # the resulting asymmetry symptom.
    rew_dof_guidance = 3.0  # was 1.0
    rew_keypos_guidance = 3.0  # was 1.0

    # guidance_temp: tried raising to 2.0 (2026-08-05) to punish the ~2x cadence mismatch found
    # between actual gait and reference more sharply, but it overshot -- rew_dof_guidance
    # collapsed to ~0.003 and flatlined there for 24k+ steps with zero movement (video confirmed
    # it looked worse, not better), while rew_keypos_guidance kept improving fine at the same
    # temp. Read as the dof error scale being large enough that exp(-2*err) crushes both the
    # reward AND its gradient near zero -- no learning signal left to climb out. Reverted to the
    # paper's 0.7, which at least keeps a live gradient (this is what took rew_dof_guidance from
    # 0.48->1.37 when weight was raised to 3.0). The cadence mismatch itself looks like it's
    # driven by something upstream of guidance_temp anyway: the reference's own achieved speed at
    # command=0.5 measured ~35% too fast in a live rollout, i.e. task reward (robot's own exact
    # speed) and guidance reward (reference's joint pattern, implicitly tuned for its own faster
    # pace) were pulling toward two different speeds. Testing that directly below by zeroing task
    # reward entirely, before touching temp again.
    guidance_temp = 0.7  # reverted from 2.0

    # NOTE: previously had a `rew_base_vel_guidance` term here rewarding the robot for matching
    # m_ref's own base_lin_vel (a stand-in task reward, back when there was no fixed command).
    # Removed now that rew_task_lin_vel (below) exists: keeping both would pull the robot toward
    # two different targets at once (m_ref's often-lower achieved speed vs. the fixed command).

    # GMP paper Table I: "Task Reward" -- Linear Velocity (weight 3.0), Angular Velocity
    # (weight 2.5): exp(-4*||v - c||^2). Previously skipped since we had no command; now that
    # a fixed command (vx, vy, wz) already exists (for the command encoder), this rewards the
    # *robot itself* for matching that same fixed target directly, instead of only indirectly
    # via m_ref (rew_base_vel_guidance above, whose own tracking of the command is imperfect).
    # task reward disabled (2026-08-05 ablation): a live rollout measured the reference's own
    # achieved speed at command=0.5 as ~0.68 m/s (35% too fast), while the robot itself (driven
    # by task reward) sat close to the true 0.5 target -- i.e. task reward and guidance reward
    # were pulling the policy toward two different speeds/cadences at once. Zeroing task reward
    # isolates whether guidance alone (temp=0.7, weight=3.0, already known not to collapse) can
    # produce tight, natural reference tracking without that conflict, before deciding whether the
    # real fix belongs in command_encoder's speed accuracy rather than in RL reward tuning.
    rew_task_lin_vel = 0.0  # was 3.0
    rew_task_ang_vel = 0.0  # was 2.5
    task_tracking_temp = 4.0
    command_wz = 0.0  # target yaw rate (0 = walk straight)

    # TEMPORARY stand-in for task reward (2026-08-05) -- MUST be reverted to rew_task_lin_vel/
    # rew_task_ang_vel once command_encoder's own speed-command sensitivity is fixed (it was found
    # to barely respond to forward_vel commands even one-step on real data -- that's the real bug,
    # this is just a workaround so RL experiments aren't blocked on it). Guidance-only training
    # (above) produces excellent joint tracking but the robot just marches in place -- nothing in
    # dof/keypos guidance depends on world-frame translation (p is base-relative, q has no
    # velocity term), so there's no incentive to actually go anywhere.
    #
    # First tried: plain-linear reward (weight * actual forward speed, no target). No ceiling, so
    # the policy just kept accelerating without bound (measured ~2.3 m/s and still climbing) while
    # dof_guidance actively got worse. Replaced with tracking the reference's OWN achieved speed
    # instead of a fixed number: bounded in [0, weight] like a real task reward, but the "target"
    # moves together with whatever guidance is already pulling the robot toward, so it can't fight
    # guidance the way the old fixed 0.5 command did (reference's own speed at that command was
    # itself measured ~35% too fast, which was the original source of the task/guidance conflict).
    rew_forward_progress = 1.5
    progress_temp = 4.0  # same shape/scale as task_tracking_temp below

    # 2026-08-05: rew_forward_progress above only compares SPEED MAGNITUDE (frame-invariant,
    # see comment on ref_speed/actual_speed in g1_gmp_env.py), so it never penalized *direction*.
    # The robot found a local optimum satisfying it via a body-frame lateral shuffle (measured
    # vx=0.11 vs vy=0.50 -- walking almost sideways relative to its own facing, not a heading/yaw
    # error) instead of walking straight ahead. Considered enabling rew_task_ang_vel (yaw-only
    # task reward) to steer this out, but that only constrains rotation rate, not lateral
    # translation -- the robot could satisfy yaw=0 perfectly while still shuffling sideways in its
    # own body frame. Directly suppressing body-frame lateral velocity is the actual fix. Same
    # temporary-workaround status as rew_forward_progress -- revert together once command_encoder
    # is fixed.
    rew_lateral_vel = -2.0

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
    """Path to a frozen command encoder checkpoint produced by
    gmp/train_command_encoder_selffeed_cmd2d.py (body-frame (forward_vel, yaw_rate) command)."""

    # fixed target command (forward_vel, yaw_rate), body-frame -- see NOTE in
    # train_command_encoder_selffeed_cmd2d.py: lateral_vel was dropped (data too sparse to
    # control reliably), so this is just the two scalars, not a 3D (vx, vy, wz) command.
    command_vx = 0.5

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
    # CVAE decoder + command encoder trained on the expanded 12-clip LAFAN1 walk dataset
    # (motions/lafan1_walk/*.npz, ~173,730 frames) instead of the original single 399-frame clip.
    # kl0.01: fixes posterior collapse seen at kl_weight=1.0. selffeed_cmd2d_w300: command
    # encoder trained via genuine multi-step auto-regressive rollout (window_length=300) against
    # the frozen decoder, conditioned on body-frame (forward_vel, yaw_rate) -- see
    # gmp/train_command_encoder_selffeed_cmd2d.py.
    gmp_checkpoint = os.path.join(GMP_CHECKPOINTS_DIR, "cvae_sweep240_kl0.01.pt")
    command_encoder_checkpoint = os.path.join(GMP_CHECKPOINTS_DIR, "command_encoder_selffeed_cmd2d_w300.pt")
