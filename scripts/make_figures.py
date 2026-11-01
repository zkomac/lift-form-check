"""
Charts for the README.

    python scripts/make_figures.py lift.mp4 --exercise rdl --start 17 --slow 0.75

Run formcheck first, it caches the pose next to the video.
"""

import argparse
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lift_form_check import bar, cleanup, exercises, i18n, metrics  # noqa: E402
from lift_form_check.config import FPS, PLATE_DIAMETER_CM, WRIST, ANKLE  # noqa: E402

matplotlib.use("Agg")

BG, PANEL, FG, GRID = "#0b0d11", "#14171d", "#e8ebf0", "#2a2f38"
LIME, CYAN, ORANGE, RED, GREY = "#c6ff3d", "#3de8ff", "#ffaa28", "#ff5046", "#a5acb8"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID,
    "axes.labelcolor": FG, "xtick.color": GREY, "ytick.color": GREY, "text.color": FG,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=14, fontweight="bold")


def bar_path_chart(t, analysis, plate_radius, start, out):
    px_per_cm = 2 * plate_radius / PLATE_DIAMETER_CM
    height = (analysis.knee_y - analysis.hub[:, 1]) / px_per_cm
    set_part = t >= start / FPS

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(t[set_part], height[set_part], color=CYAN, lw=2.2, label="bar (plate hub)")
    ax.axhline(0, color=ORANGE, ls="--", lw=1.4, label="knee level")

    for n, rep in enumerate(analysis.reps, 1):
        warned = any(kind == "warn" for kind, *_ in rep["flags"])
        x, y = t[rep["frame"]], height[rep["frame"]]
        ax.scatter(x, y, color=RED if warned else LIME, s=60, zorder=3)
        ax.annotate(str(n), (x, y), xytext=(0, -16), textcoords="offset points", ha="center")

    ax.set_xlabel("video time [s]")
    ax.set_ylabel("bar height vs knee [cm]")
    title(ax, "Bar path through the set")
    ax.legend(facecolor=PANEL, edgecolor=GRID, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)


def reps_chart(analysis, profile, out):
    reps = analysis.reps
    numbers = np.arange(1, len(reps) + 1)
    columns = profile["columns"]

    fig, axes = plt.subplots(1, len(columns), figsize=(12, 3.6), dpi=150)
    for ax, (key, _, label) in zip(axes, columns):
        values = [rep[key] for rep in reps]
        colors = [RED if any(m == key and k == "warn" for k, m, *_ in rep["flags"]) else LIME
                  for rep in reps]
        ax.bar(numbers, values, color=colors, width=0.6)
        ax.set_title(i18n.pick(label).capitalize(), fontsize=11, loc="left")
        ax.set_xticks(numbers)
        ax.set_xlabel("rep")

    fig.suptitle("Metrics per rep", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)


def cleanup_chart(t, raw_pose, pts, start, out):
    raw = np.array([p[ANKLE][1] if p else np.nan for p in raw_pose])
    set_part = t >= start / FPS

    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
    ax.plot(t[set_part], raw[set_part], color=RED, lw=1, alpha=0.8, label="ankle – raw detection")
    ax.plot(t[set_part], pts[set_part, ANKLE, 1], color=LIME, lw=2.4, label="cleaned (Hampel + feet lock)")
    ax.invert_yaxis()
    ax.set_xlabel("video time [s]")
    ax.set_ylabel("y [px]")
    title(ax, "Keypoint cleanup: the plate hides the ankle")
    ax.legend(facecolor=PANEL, edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(out)


def pipeline_chart(out):
    steps = [
        ("Video", "side view\n1080×1920, 30 fps", GREY),
        ("Pose", "MediaPipe\n33 keypoints / frame", CYAN),
        ("Cleanup", "pandas Hampel filter\nfeet lock", CYAN),
        ("Bar", "OpenCV Hough +\ntemplate matching", CYAN),
        ("Metrics", "angles, depth,\ntempo, reps", LIME),
        ("Render", "9:16 overlay\n+ warnings", LIME),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.6), dpi=150)
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.6)

    for n, (name, text, color) in enumerate(steps):
        x = 0.15 + n * 2
        ax.add_patch(FancyBboxPatch((x, 0.5), 1.7, 1.6, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=PANEL, ec=color, lw=2))
        ax.text(x + 0.85, 1.65, name, ha="center", va="center", fontsize=13, fontweight="bold", color=color)
        ax.text(x + 0.85, 1.0, text, ha="center", va="center", fontsize=9, color=FG)
        if n < len(steps) - 1:
            ax.annotate("", xy=(x + 2.0, 1.3), xytext=(x + 1.75, 1.3),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6))

    fig.savefig(out, bbox_inches="tight", pad_inches=0.1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--exercise", default="rdl")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--slow", type=float, default=1.0)
    p.add_argument("--out", default="docs")
    args = p.parse_args()

    profile = exercises.get(args.exercise)
    start = int(args.start * FPS)
    raw_pose = json.load(open(os.path.splitext(args.video)[0] + "_pose.json"))

    positions, radius = bar.track(args.video, start, raw_pose[start][WRIST][:2])
    pts = cleanup.clean(raw_pose, start, profile["lock_knees"])
    hub, score = cleanup.clean_bar(positions)
    analysis = metrics.analyse(pts, hub, score, start, radius, args.slow, profile)
    t = np.arange(len(pts)) / FPS

    os.makedirs(args.out, exist_ok=True)
    bar_path_chart(t, analysis, radius, start, f"{args.out}/bar_path.png")
    reps_chart(analysis, profile, f"{args.out}/reps.png")
    cleanup_chart(t, raw_pose, pts, start, f"{args.out}/cleanup.png")
    pipeline_chart(f"{args.out}/pipeline.png")
    print("saved to", args.out)


if __name__ == "__main__":
    main()
