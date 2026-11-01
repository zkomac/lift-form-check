"""
Cleaning up the keypoints.

The plates hide knees and ankles for most of the set, and MediaPipe then
guesses them way off (the ankle jumps up to 25 cm). Without this the leg
lines jump around all the time.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import KNEE

FEET = range(27, 33)        # ankles, heels, toes
KNEES = (KNEE, KNEE + 1)


def to_array(pose):
    """List of frames -> numpy array (frames x 33 x 2). Missing frames reuse the last one."""
    last = next(p for p in pose if p)
    rows = []
    for p in pose:
        last = p or last
        rows.append([point[:2] for point in last])
    return np.array(rows, float)


def hampel(series, window=21, max_jump=35):
    """Replace points that jump away from the rolling median."""
    df = pd.DataFrame(series, columns=["x", "y"])
    median = df.rolling(window, center=True, min_periods=1).median()
    jump = (df - median).abs().max(axis=1)
    df[jump > max_jump] = median[jump > max_jump]
    return df.to_numpy()


def smooth(series, window=7):
    """Savitzky-Golay keeps the shape of the movement better than a plain average."""
    return savgol_filter(series, window, polyorder=2, axis=0, mode="nearest")


def clean(pose, start, lock_knees=False):
    pts = to_array(pose)

    for k in range(33):
        limit = 35 if k >= 23 else 60   # legs are hidden more often
        pts[:, k] = hampel(pts[:, k], max_jump=limit)

    # feet don't move in a barbell lift, so lock them once the set starts
    for k in FEET:
        pts[start:, k] = np.median(pts[start:, k], axis=0)

    out = smooth(pts)

    # RDL: knees only travel a few cm. keep a little motion around the median
    if lock_knees:
        for k in KNEES:
            center = np.median(pts[start:, k], axis=0)
            offset = pd.DataFrame(pts[start:, k] - center)
            offset = offset.rolling(31, center=True, min_periods=1).median().clip(-25, 25)
            offset = offset.rolling(15, center=True, min_periods=1).mean()
            out[start:, k] = center + offset.to_numpy()

    return out


def clean_bar(positions):
    """Fill frames before tracking started and smooth the hub path."""
    first = next(p for p in positions if p)
    xy = np.array([p[:2] if p else first[:2] for p in positions], float)
    score = np.array([p[2] if p else np.nan for p in positions])
    return smooth(xy, 5), score
