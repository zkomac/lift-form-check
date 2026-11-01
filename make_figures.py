"""Charts for the README. Run after formcheck.py (uses the cached pose + video).

    python make_figures.py video.mp4 <start s> <slow> <exercise>
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import formcheck as fc

video, start, slow, ex = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
s0 = int(start * fc.FPS)
pose = json.load(open(video.rsplit(".", 1)[0] + "_pose.json"))
bar, r = fc.track_bar(video, s0, pose[s0][15][:2])
pts = fc.clean_pose(pose, s0, fc.EXERCISES[ex]["lock_knees"])
m = fc.analyse(pts, bar, s0, r, slow, ex)
reps = m["reps"]

BG, PANEL, FG, GRID = "#0b0d11", "#14171d", "#e8ebf0", "#2a2f38"
LIME, CYAN, ORANGE, RED, GREY = "#c6ff3d", "#3de8ff", "#ffaa28", "#ff5046", "#a5acb8"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID,
                     "axes.labelcolor": FG, "xtick.color": GREY, "ytick.color": GREY, "text.color": FG,
                     "font.family": "DejaVu Sans", "font.size": 11, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6})
t = np.arange(len(pts)) / fc.FPS

# 1) bar path over time
fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
px_cm = 2 * r / fc.PLATE_CM
y = (m["knee_y"] - m["hub"][:, 1]) / px_cm   # cm above knee (+) / below (-)
sel = t >= start
ax.plot(t[sel], y[sel], color=CYAN, lw=2.2, label="drog (sredina plošče)")
ax.axhline(0, color=ORANGE, ls="--", lw=1.4, label="nivo kolena")
for i, rp in enumerate(reps, 1):
    bad = any(f[0] == "warn" for f in rp["flags"])
    ax.scatter(t[rp["frame"]], y[rp["frame"]], color=RED if bad else LIME, s=60, zorder=3)
    ax.annotate(f"{i}", (t[rp["frame"]], y[rp["frame"]]), xytext=(0, -16), textcoords="offset points",
                ha="center", color=FG, fontsize=10)
ax.set_xlabel("čas videa [s]")
ax.set_ylabel("višina droga glede na koleno [cm]")
ax.set_title("Pot droga skozi set", loc="left", fontsize=14, fontweight="bold")
ax.legend(facecolor=PANEL, edgecolor=GRID, loc="upper right")
fig.tight_layout()
fig.savefig("docs/bar_path.png")

# 2) per rep metrics
fig, axs = plt.subplots(1, 4, figsize=(12, 3.6), dpi=150)
n = np.arange(1, len(reps) + 1)
for ax, (title, key, _) in zip(axs, fc.EXERCISES[ex]["cols"]):
    vals = [r_[key] for r_ in reps]
    cols = [RED if any(fc.FLAG_COL.get(f[1]) == key and f[0] == "warn" for f in r_["flags"]) else LIME for r_ in reps]
    ax.bar(n, vals, color=cols, width=0.6)
    ax.set_title(title.capitalize(), fontsize=11, loc="left")
    ax.set_xticks(n)
    ax.set_xlabel("ponovitev")
fig.suptitle("Metrike po ponovitvah", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("docs/reps.png")

# 3) why the keypoint cleanup matters (ankle hidden behind the plate)
raw = np.array([p[27][1] if p else np.nan for p in pose])
fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
ax.plot(t[sel], raw[sel], color=RED, lw=1, alpha=0.8, label="gleženj – surova zaznava")
ax.plot(t[sel], pts[sel, 27, 1], color=LIME, lw=2.4, label="po čiščenju (Hampel + zaklep stopal)")
ax.invert_yaxis()
ax.set_ylabel("y [px]")
ax.set_xlabel("čas videa [s]")
ax.set_title("Čiščenje točk: plošča zakriva gleženj", loc="left", fontsize=14, fontweight="bold")
ax.legend(facecolor=PANEL, edgecolor=GRID)
fig.tight_layout()
fig.savefig("docs/cleanup.png")

# 4) pipeline diagram
fig, ax = plt.subplots(figsize=(12, 2.6), dpi=150)
ax.set_axis_off()
ax.set_xlim(0, 12); ax.set_ylim(0, 2.6)
steps = [("Video", "stranski posnetek\n1080×1920, 30 fps", GREY),
         ("Pose", "MediaPipe\n33 točk / sličico", CYAN),
         ("Čiščenje", "Hampel filter\nzaklep stopal", CYAN),
         ("Drog", "template matching\nsredine plošče", CYAN),
         ("Metrike", "koti, globina,\ntempo, ponovitve", LIME),
         ("Render", "overlay 9:16\n+ opozorila", LIME)]
for i, (title, sub, c) in enumerate(steps):
    x = 0.15 + i * 2
    ax.add_patch(FancyBboxPatch((x, 0.5), 1.7, 1.6, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=PANEL, ec=c, lw=2))
    ax.text(x + 0.85, 1.65, title, ha="center", va="center", fontsize=13, fontweight="bold", color=c)
    ax.text(x + 0.85, 1.0, sub, ha="center", va="center", fontsize=9, color=FG)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 2.0, 1.3), xytext=(x + 1.75, 1.3),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6))
fig.savefig("docs/pipeline.png", bbox_inches="tight", pad_inches=0.1)
print("figures ok")
