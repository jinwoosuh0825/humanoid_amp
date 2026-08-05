# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Loads a trained GMP policy checkpoint and, instead of just recording video, logs the actual
robot joint positions vs. the online-generated GMP reference (m_ref) side by side for one
episode, so we can check numerically (not just visually) how closely the trained policy is
actually tracking the reference -- e.g. whether left/right leg joints track the reference with
similar fidelity, or whether one leg diverges from its reference counterpart more than the other.

Usage:
    python eval_reference_tracking.py --task Isaac-G1-GMP-Walk-Direct-v0 \
        --checkpoint logs/skrl/g1_gmp_walk/<run>/checkpoints/agent_30300.pt \
        --num_envs 1 --steps 300 --output /tmp/ref_tracking.npz
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--algorithm", type=str, default="PPO")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import torch
import gymnasium as gym
import skrl
from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper

agent_cfg_entry_point = "skrl_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, experiment_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    experiment_cfg["seed"] = args_cli.seed

    # explicit, belt-and-suspenders seeding right before env construction/reset -- setting
    # env_cfg.seed alone was found to NOT actually change the sampled reset frame or z-noise
    # (4 different --seed values all produced the exact same reference trajectory), so force it
    # here instead of trusting IsaacLab/skrl's internal seeding plumbing to pick it up.
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env_wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env_wrapped, experiment_cfg)
    print(f"[EVAL] Loading checkpoint: {args_cli.checkpoint}")
    runner.agent.load(os.path.abspath(args_cli.checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    # re-seed again right before reset: Runner/agent construction and checkpoint loading above
    # may themselves have consumed/reset RNG state.
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)

    base_env = env.unwrapped
    dof_names = base_env.robot.data.joint_names
    motion_dof_indexes = base_env.motion_dof_indexes
    motion_state_spec = base_env.motion_state_spec

    obs, _ = env_wrapped.reset()
    states = env_wrapped.state()

    N = args_cli.num_envs
    q_actual_log = np.zeros((args_cli.steps, N, len(dof_names)), dtype=np.float32)
    q_ref_log = np.zeros((args_cli.steps, N, len(dof_names)), dtype=np.float32)
    # base_lin_vel/base_ang_vel: actual is measured directly from the robot (body frame, same
    # convention as _compute_base_frame_velocities in g1_gmp_env.py); reference is read straight
    # out of m_ref's own (base_lin_vel, base_ang_vel) components -- no extra computation needed,
    # since that's literally what the first 6 dims of the GMP motion-state m_t already are.
    v_actual_log = np.zeros((args_cli.steps, N, 3), dtype=np.float32)
    w_actual_log = np.zeros((args_cli.steps, N, 3), dtype=np.float32)
    v_ref_log = np.zeros((args_cli.steps, N, 3), dtype=np.float32)
    w_ref_log = np.zeros((args_cli.steps, N, 3), dtype=np.float32)

    with torch.inference_mode():
        for t in range(args_cli.steps):
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])

            q_actual_log[t] = base_env.robot.data.joint_pos.cpu().numpy()
            ref = motion_state_spec.split(base_env.m_ref)
            q_ref_log[t] = ref["dof_pos"][:, motion_dof_indexes].cpu().numpy()
            v_ref_log[t] = ref["base_lin_vel"].cpu().numpy()
            w_ref_log[t] = ref["base_ang_vel"].cpu().numpy()
            v_actual_b, w_actual_b = base_env._compute_base_frame_velocities()
            v_actual_log[t] = v_actual_b.cpu().numpy()
            w_actual_log[t] = w_actual_b.cpu().numpy()

            obs, _, _, _, _ = env_wrapped.step(actions)
            states = env_wrapped.state()

    np.savez(
        args_cli.output,
        dof_names=np.array(dof_names, dtype=np.str_),
        q_actual=q_actual_log,
        q_ref=q_ref_log,
        v_actual=v_actual_log,
        w_actual=w_actual_log,
        v_ref=v_ref_log,
        w_ref=w_ref_log,
    )
    print(f"[EVAL] Saved {args_cli.output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
