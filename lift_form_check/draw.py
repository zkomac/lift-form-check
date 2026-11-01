"""Small drawing helpers used by the overlay and the HUD."""

import cv2
import numpy as np
from PIL import ImageFont

from .config import PANEL

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"


def font(name, size):
    try:
        return ImageFont.truetype(FONT_DIR + name, size)
    except OSError:
        return ImageFont.load_default(size)


def point(p):
    return int(p[0]), int(p[1])


def ease(t):
    """0..1 -> 0..1 with a soft end, used for fades and slides."""
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


# ---- OpenCV (lines on the video layer)

def dashed_line(img, a, b, color, thickness=2, dash=16, gap=10):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    length = np.linalg.norm(b - a)
    step = (b - a) / max(length, 1e-6)

    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        cv2.line(img, point(a + step * pos), point(a + step * end), color, thickness, cv2.LINE_AA)
        pos += dash + gap


def glow_line(layer, glow, a, b, color, thickness):
    cv2.line(glow, point(a), point(b), color, thickness * 4, cv2.LINE_AA)
    cv2.line(layer, point(a), point(b), color, thickness, cv2.LINE_AA)


def joint(layer, p, color, radius):
    cv2.circle(layer, point(p), radius + 5, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.circle(layer, point(p), radius, color, -1, cv2.LINE_AA)


# ---- PIL (text and panels)

def tag(draw, xy, text, fnt, color, anchor="lm"):
    """Text on a solid colored label."""
    x0, y0, x1, y1 = draw.textbbox(xy, text, font=fnt, anchor=anchor)
    draw.rectangle((x0 - 8, y0 - 5, x1 + 8, y1 + 5), fill=(*color, 230))
    draw.text(xy, text, font=fnt, fill=PANEL, anchor=anchor)


def panel(draw, box, outline=None, alpha=165):
    draw.rounded_rectangle(
        box, radius=20, fill=(*PANEL, alpha),
        outline=outline, width=2 if outline else 0,
    )
