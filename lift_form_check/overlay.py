"""
Everything drawn on top of the video itself: skeleton, angle arc,
bar path and the 'detection boxes'.
"""

import cv2
import numpy as np
import supervision as sv

from . import draw
from .config import (WIDTH, SAFE_TOP, SAFE_BOTTOM, LIME, CYAN, WHITE, ORANGE, RED, GREY,
                     EAR, SHOULDER, HIP, KNEE, ANKLE)

# mediapipe body connections (without the face and hands)
SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32), (7, 11), (8, 12),
]
DOTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
TRAIL = 75        # bar path length in frames (2.5 s)
NECK_LIMIT = 12   # degrees, neck turns red above this


def sv_color(rgb):
    # our frames are RGB, supervision draws in BGR -> swap the channels
    return sv.Color(r=rgb[2], g=rgb[1], b=rgb[0])


# supervision wants 1-based edges over the points we pass in (only DOTS)
_index = {k: n + 1 for n, k in enumerate(DOTS)}
MESH_EDGES = [(_index[a], _index[b]) for a, b in SKELETON if a in _index and b in _index]

mesh_lines = sv.EdgeAnnotator(color=sv_color((200, 200, 200)), thickness=1, edges=MESH_EDGES)
mesh_dots = sv.VertexAnnotator(color=sv_color((230, 230, 230)), radius=4)
person_corners = sv.BoxCornerAnnotator(color=sv_color(LIME), thickness=3, corner_length=34)
bar_corners = sv.BoxCornerAnnotator(color=sv_color(CYAN), thickness=3, corner_length=26)


def box(x0, y0, x1, y1):
    return sv.Detections(xyxy=np.array([[x0, y0, x1, y1]], float), class_id=np.array([0]))


def mesh(layer, kp):
    """Thin full-body mesh, always visible."""
    points = sv.KeyPoints(xy=kp[DOTS][None].astype(np.float32))
    mesh_lines.annotate(layer, points)
    mesh_dots.annotate(layer, points)


def person_box(layer, kp):
    x0, y0 = kp[:, 0].min() - 40, kp[:, 1].min() - 60
    x1, y1 = kp[:, 0].max() + 40, kp[:, 1].max() + 30
    person_corners.annotate(layer, box(x0, y0, x1, y1))
    return ("PERSON 0.98 · ID 01", (x0, y0 - 14), LIME)


def reference_lines(layer, a):
    draw.dashed_line(layer, (a.midfoot, SAFE_TOP + 380), (a.midfoot, SAFE_BOTTOM - 255), GREY)
    draw.dashed_line(layer, (60, a.knee_y), (WIDTH - 60, a.knee_y), ORANGE, 2, 12, 9)


def body_chain(layer, glow, kp, neck_angle):
    shoulder, hip, knee, ankle, ear = kp[SHOULDER], kp[HIP], kp[KNEE], kp[ANKLE], kp[EAR]
    back = (shoulder - hip) / np.linalg.norm(shoulder - hip)
    neck_color = RED if neck_angle > NECK_LIMIT else (230, 230, 230)

    draw.glow_line(layer, glow, hip - back * 50, shoulder + back * 100, LIME, 6)
    draw.glow_line(layer, glow, hip, knee, CYAN, 5)
    draw.glow_line(layer, glow, knee, ankle, CYAN, 5)
    draw.glow_line(layer, glow, shoulder, ear, neck_color, 4)

    draw.joint(layer, ear, neck_color, 9)
    draw.joint(layer, shoulder, LIME, 12)
    draw.joint(layer, hip, LIME, 14)
    draw.joint(layer, knee, CYAN, 12)
    draw.joint(layer, ankle, CYAN, 10)


def hip_arc(layer, kp, radius=66):
    """Arc between torso and thigh. Returns where to put the angle label."""
    hip = kp[HIP]
    a1 = np.degrees(np.arctan2(*(kp[SHOULDER] - hip)[::-1]))
    a2 = np.degrees(np.arctan2(*(kp[KNEE] - hip)[::-1]))
    sweep = (a2 - a1 + 540) % 360 - 180

    cv2.ellipse(layer, draw.point(hip), (radius, radius), 0, a1, a1 + sweep, LIME, 4, cv2.LINE_AA)

    middle = np.radians(a1 + sweep / 2)
    return hip[0] + np.cos(middle) * (radius + 52), hip[1] + np.sin(middle) * (radius + 52)


def bar_path(layer, glow, hub, i, start):
    trail = hub[max(start, i - TRAIL):i + 1]
    for j in range(1, len(trail)):
        fade = j / len(trail)
        color = tuple(int(c * fade) for c in CYAN)
        cv2.line(glow, draw.point(trail[j - 1]), draw.point(trail[j]), color, 14, cv2.LINE_AA)
        cv2.line(layer, draw.point(trail[j - 1]), draw.point(trail[j]), color, 5, cv2.LINE_AA)


def bar_box(layer, center, radius, score):
    x, y = center
    cv2.rectangle(layer, draw.point((x - radius, y - radius)), draw.point((x + radius, y + radius)),
                  CYAN, 1, cv2.LINE_AA)
    bar_corners.annotate(layer, box(x - radius, y - radius, x + radius, y + radius))
    cv2.drawMarker(layer, draw.point(center), WHITE, cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)

    if np.isnan(score):
        score = 0.9
    return (f"BARBELL {score:.2f} · TRK", (x - radius, y + radius + 22), CYAN)


def render(frame, i, pts, a, start, plate_radius):
    """
    Draws on a copy of the frame. Returns (frame, labels for the HUD, hip label position).
    The analysis part fades in when the set starts.
    """
    live = i >= start
    strength = draw.ease((i - start) / 20) if live else 0.0

    layer = np.zeros_like(frame)
    glow = np.zeros_like(frame)
    kp = pts[i]

    mesh(layer, kp)
    labels = [person_box(layer, kp)]
    arc_label = None

    if live:
        reference_lines(layer, a)
        body_chain(layer, glow, kp, a.neck[i])
        arc_label = hip_arc(layer, kp)
        bar_path(layer, glow, a.hub, i, start)
        labels.append(bar_box(layer, a.hub[i], plate_radius, a.score[i]))

    # darken the video a bit so the lines pop, then add glow and lines
    out = frame * (1 - 0.22 * max(strength, 0.4))
    glow = cv2.GaussianBlur(glow, (0, 0), 9)
    out = np.clip(out + glow * 0.9 * strength, 0, 255)

    mask = layer.any(axis=2)
    out[mask] = out[mask] * 0.15 + layer[mask] * 0.85

    return out.astype(np.uint8), labels, arc_label
