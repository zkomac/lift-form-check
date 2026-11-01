"""Joint angles, rep detection and per-rep numbers."""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from . import exercises
from .config import FPS, PLATE_DIAMETER_CM, EAR, SHOULDER, HIP, KNEE, ANKLE, TOES


def joint_angle(a, b, c):
    """Angle at point b (degrees) for every frame."""
    v1 = a - b
    v2 = c - b
    cos = (v1 * v2).sum(axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def signed_angle(u, v):
    """Angle from u to v, positive = counter-clockwise on screen."""
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dot = (u * v).sum(axis=1)
    return np.degrees(np.arctan2(cross, dot))


@dataclass
class Analysis:
    hub: np.ndarray            # bar position per frame (x, y)
    score: np.ndarray          # tracker confidence per frame
    hip: np.ndarray            # hip angle
    knee: np.ndarray           # knee angle
    torso: np.ndarray          # torso lean from horizontal
    neck: np.ndarray           # head vs back line, + = extension
    knee_y: float              # knee height during the set (px)
    midfoot: float             # x of mid foot (px)
    reps: list = field(default_factory=list)


def body_angles(pts):
    shoulder, hip, knee, ankle, ear = (pts[:, k] for k in (SHOULDER, HIP, KNEE, ANKLE, EAR))

    torso_vec = shoulder - hip
    torso = np.degrees(np.arctan2(-torso_vec[:, 1], np.abs(torso_vec[:, 0])))
    neck = signed_angle(shoulder - hip, ear - shoulder)

    return dict(
        hip=joint_angle(shoulder, hip, knee),
        knee=joint_angle(hip, knee, ankle),
        torso=torso,
        neck=neck,
    )


def find_turns(bar_y, start, slow):
    """Lowest (bottoms) and highest (tops) bar points, at least ~1.3 s apart."""
    distance = int(1.3 * FPS / slow)
    bottoms, _ = find_peaks(bar_y, prominence=120, distance=distance)
    tops, _ = find_peaks(-bar_y, prominence=120, distance=distance)
    bottoms = [int(b) for b in bottoms if b > start]
    tops = [int(t) for t in tops if t > start - FPS]
    return bottoms, tops


def reps_at_bottom(a, pts, bottoms, tops, start, px_per_cm, slow, depth_ref):
    """RDL / squat: a rep is the lowest point of the bar."""
    reps = []
    for b in bottoms:
        top = max([t for t in tops if t < b], default=start)

        if depth_ref == "hip":
            depth = pts[b, HIP, 1] - pts[b, KNEE, 1]
        else:
            depth = a.hub[b, 1] - a.knee_y

        reps.append(dict(
            frame=b,
            depth=float(depth / px_per_cm),
            hip=float(a.hip[b]),
            knee=float(a.knee[b]),
            neck=float(a.neck[b]),
            down=(b - top) / FPS * slow,   # real seconds
        ))
    return reps


def reps_at_top(a, pts, bottoms, tops, start, slow):
    """Deadlift: a rep goes from the floor to lockout."""
    reps = []
    for t in tops:
        if t <= start:
            continue
        b = max([x for x in bottoms if x < t], default=start)

        # first 40 % of the pull: do the hips rise faster than the bar?
        early = b + max(2, int((t - b) * 0.4))
        hip_rise = pts[b, HIP, 1] - pts[early, HIP, 1]
        bar_rise = max(a.hub[b, 1] - a.hub[early, 1], 1)

        reps.append(dict(
            frame=t,
            lockout=float(a.hip[max(t - 5, 0):t + 6].max()),
            hips=float(hip_rise / bar_rise),
            neck=float(a.neck[b]),
            hip=float(a.hip[t]),
            up=(t - b) / FPS * slow,
        ))
    return reps


def analyse(pts, hub, score, start, plate_radius, slow, profile):
    angles = body_angles(pts)
    feet_x = (pts[:, ANKLE, 0] + pts[:, TOES, 0]) / 2

    a = Analysis(
        hub=hub,
        score=score,
        knee_y=float(np.median(pts[start:, KNEE, 1])),
        midfoot=float(np.median(feet_x[start:])),
        **angles,
    )

    px_per_cm = 2 * plate_radius / PLATE_DIAMETER_CM
    bottoms, tops = find_turns(hub[:, 1], start, slow)

    if profile["count"] == "top":
        a.reps = reps_at_top(a, pts, bottoms, tops, start, slow)
    else:
        a.reps = reps_at_bottom(a, pts, bottoms, tops, start, px_per_cm, slow, profile["depth_ref"])

    for rep in a.reps:
        rep["flags"] = exercises.check(rep, profile)

    return a
