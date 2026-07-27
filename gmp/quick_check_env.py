# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quick standalone sanity check for the GMP env: steps with random actions and prints reward stats."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import humanoid_amp  # noqa: F401  (registers gym envs)

env_cfg_cls = __import__("humanoid_amp.g1_gmp_env_cfg", fromlist=["G1GmpWalkEnvCfg"]).G1GmpWalkEnvCfg
env_cfg = env_cfg_cls()
env_cfg.scene.num_envs = args_cli.num_envs

env = gym.make("Isaac-G1-GMP-Walk-Direct-v0", cfg=env_cfg)
obs, _ = env.reset()
print(f"[CHECK] obs shape: {obs['policy'].shape}")

for step in range(args_cli.steps):
    actions = torch.zeros(env.unwrapped.num_envs, env.unwrapped.cfg.action_space, device=env.unwrapped.device)
    obs, reward, terminated, truncated, extras = env.step(actions)
    log = extras.get("log", {})
    log_str = ", ".join(f"{k}={v.item():.4f}" for k, v in log.items())
    print(
        f"[CHECK] step {step:2d} reward mean={reward.mean().item():.4f} "
        f"min={reward.min().item():.4f} max={reward.max().item():.4f} nan={torch.isnan(reward).any().item()} "
        f"| {log_str}"
    )

env.close()
simulation_app.close()
