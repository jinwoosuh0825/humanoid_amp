# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Generative Motion Prior (GMP): CVAE-based motion guidance for AMP-style humanoid locomotion,
following "Natural Humanoid Robot Locomotion with Generative Motion Prior" (Zhang et al., IROS 2025).
"""

from .cvae_model import MotionDecoder, MotionEncoder, MotionCVAE
from .motion_state import MotionState, build_motion_state_sequence
