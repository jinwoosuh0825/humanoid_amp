# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Headless version of motions/motion_viewer.py: renders the same 3D skeleton animation but saves
it to a GIF file instead of calling plt.show() (which needs an interactive display).

Usage:
    python render_motion_video.py --motion_file ../motions/cvae_rollout_check.npz \
        --output ../motions/cvae_rollout_check.gif --render_scene
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions"))
from motion_loader import MotionLoader  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--render_scene", action="store_true")
    parser.add_argument("--stride", type=int, default=2, help="Render every N-th frame (speed up / shrink file).")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--fixed_extent",
        type=float,
        default=None,
        help="Chase camera: fixed half-width (meters) of the view box, always centered on the "
        "pelvis. Use the SAME value across multiple renders to make them directly comparable "
        "(same zoom level), and to keep the character large/visible instead of shrinking to a "
        "dot inside a box sized to fit its entire (possibly long) path.",
    )
    args = parser.parse_args()

    loader = MotionLoader(motion_file=args.motion_file, device="cpu")
    body_positions = loader.body_positions.cpu().numpy()[:: args.stride]
    num_frames = body_positions.shape[0]

    print("Body position ranges:")
    for i, name in enumerate(loader.body_names):
        lo = np.min(body_positions[:, i], axis=0).round(2)
        hi = np.max(body_positions[:, i], axis=0).round(2)
        print(f"  |-- [{name}] min: {lo}, max: {hi}")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    state = {"frame": 0}

    def update(_):
        frame = state["frame"]
        vertices = body_positions[frame]
        ax.clear()
        ax.scatter(*vertices.T, color="black", depthshade=False)
        if args.fixed_extent is not None:
            # chase camera: fixed-size box, always centered on the pelvis (body index 0) --
            # keeps the character a consistent, visible size regardless of how far it travels,
            # and (with the same --fixed_extent value) makes separate renders directly comparable.
            center = vertices[0].copy()
            diff = np.array([args.fixed_extent] * 3)
        elif args.render_scene:
            minimum = np.min(body_positions.reshape(-1, 3), axis=0)
            maximum = np.max(body_positions.reshape(-1, 3), axis=0)
            center = 0.5 * (maximum + minimum)
            # cube-shaped view box (largest single-axis extent applied to all 3 axes), instead of
            # stretching per-axis to the exact trajectory extent -- a straight/long-forward path
            # with little lateral movement would otherwise render as a thin, hard-to-read corridor.
            diff = np.array([0.75 * np.max(maximum - minimum).item()] * 3)
        else:
            minimum = np.min(vertices, axis=0)
            maximum = np.max(vertices, axis=0)
            center = 0.5 * (maximum + minimum)
            diff = np.array([0.75 * np.max(maximum - minimum).item()] * 3)
        diff = np.maximum(diff, 1e-3)
        ax.set_xlim((center[0] - diff[0], center[0] + diff[0]))
        ax.set_ylim((center[1] - diff[1], center[1] + diff[1]))
        ax.set_zlim((center[2] - diff[2], center[2] + diff[2]))
        ax.set_box_aspect(aspect=diff / diff[0])
        x, y = np.meshgrid([center[0] - diff[0], center[0] + diff[0]], [center[1] - diff[1], center[1] + diff[1]])
        ax.plot_surface(x, y, np.zeros_like(x), color="green", alpha=0.2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"frame: {frame}/{num_frames}")
        state["frame"] = (frame + 1) % num_frames

    anim = matplotlib.animation.FuncAnimation(fig=fig, func=update, frames=num_frames, interval=1000 / args.fps)
    anim.save(args.output, writer=matplotlib.animation.PillowWriter(fps=args.fps))
    print(f"[GMP] Saved animation ({num_frames} frames) to {args.output}")


if __name__ == "__main__":
    main()
