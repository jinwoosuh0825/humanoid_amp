"""
Patches skrl's installed AMP agent (skrl/agents/torch/amp/amp.py) so that the style reward
(computed inside `update()` from the discriminator, once per rollout) is actually logged --
vanilla skrl never tracks it: `record_transition()` only logs the raw per-step environment
reward (our small regularization penalties, since task_reward_scale=0 in our config), and
`update()` computes style_reward purely as a local variable used for GAE, never passed to
`track_data()` or saved anywhere.

This patch adds, right after style_reward is computed:
  1. `self.track_data(...)` calls so "Reward / Style reward (mean/std)" show up in TensorBoard.
  2. An append-only CSV dump (`<experiment_dir>/style_reward_log.csv`) independent of
     TensorBoard, so the curve can be plotted with matplotlib even if TensorBoard rendering is
     unreliable.

Idempotent: safe to run multiple times (checks for the marker comment first).

Usage:
    python apply_amp_style_reward_patch.py
"""

import skrl.agents.torch.amp.amp as amp_module

TARGET_FILE = amp_module.__file__

MARKER = "# --- GMP patch: log style reward (not tracked by vanilla skrl) ---"

OLD = """            style_reward = -torch.log(
                torch.maximum(1 - 1 / (1 + torch.exp(-amp_logits)), torch.tensor(0.0001, device=self.device))
            ).view(rewards.shape)

        combined_rewards = self.cfg.task_reward_scale * rewards + self.cfg.style_reward_scale * style_reward
"""

NEW = """            style_reward = -torch.log(
                torch.maximum(1 - 1 / (1 + torch.exp(-amp_logits)), torch.tensor(0.0001, device=self.device))
            ).view(rewards.shape)

        combined_rewards = self.cfg.task_reward_scale * rewards + self.cfg.style_reward_scale * style_reward

        # --- GMP patch: log style reward (not tracked by vanilla skrl) ---
        self.track_data("Reward / Style reward (mean)", style_reward.mean().item())
        self.track_data("Reward / Style reward (std)", style_reward.std().item())
        os.makedirs(self.experiment_dir, exist_ok=True)
        _style_log_path = os.path.join(self.experiment_dir, "style_reward_log.csv")
        _write_header = not os.path.exists(_style_log_path)
        with open(_style_log_path, "a") as _f:
            if _write_header:
                _f.write("timestep,style_reward_mean,style_reward_std\\n")
            _f.write(f"{timestep},{style_reward.mean().item()},{style_reward.std().item()}\\n")
        # --- end GMP patch ---
"""

with open(TARGET_FILE, "r") as f:
    content = f.read()

if MARKER in content:
    print(f"[GMP patch] Already applied to {TARGET_FILE}, nothing to do.")
else:
    if OLD not in content:
        raise RuntimeError(
            "Expected code block not found -- skrl's amp.py may have changed. "
            "Inspect the file manually before patching."
        )
    content = content.replace(OLD, NEW, 1)
    if "\nimport os\n" not in content and not content.startswith("import os\n"):
        content = content.replace("import itertools\n", "import itertools\nimport os\n", 1)
    with open(TARGET_FILE, "w") as f:
        f.write(content)
    print(f"[GMP patch] Applied to {TARGET_FILE}")
