"""Finding and following the barbell (the plate hub)."""

import cv2
import numpy as np
from tqdm import tqdm

from . import video
from .config import WIDTH, HEIGHT


def find_plate(gray, hand):
    """
    Hough circle search on one frame. Running it on every frame is way too slow,
    so it's only used to find the plate at the start. The bar is in the hands,
    so the circle closest to the wrist wins.
    """
    small = cv2.resize(gray, (WIDTH // 2, HEIGHT // 2))
    small = cv2.medianBlur(small, 5)

    circles = cv2.HoughCircles(
        small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=60,
        param1=80, param2=35, minRadius=50, maxRadius=110,
    )
    if circles is None:
        raise SystemExit("could not find the plate - is the bar visible at --start?")

    circles = circles[0] * 2  # back to full size
    x, y, r = min(circles, key=lambda c: np.hypot(c[0] - hand[0], c[1] - hand[1]))
    return int(x), int(y), int(r)


class HubTracker:
    """Template matching on the metal sleeve in the middle of the plate."""

    HALF = 36     # template is 72 x 72 px
    SEARCH = 90   # how far the hub can move between two frames

    def __init__(self, gray, x, y):
        # the sleeve is the brightest spot near the circle centre
        roi = gray[y - 60:y + 60, x - 60:x + 60]
        roi = cv2.GaussianBlur(roi, (15, 15), 0)
        _, _, _, bright = cv2.minMaxLoc(roi)

        self.x = x - 60 + bright[0]
        self.y = y - 60 + bright[1]
        self.template = self._patch(gray).astype(np.float32)

    def _patch(self, gray):
        h = self.HALF
        return gray[self.y - h:self.y + h, self.x - h:self.x + h]

    def update(self, gray):
        h, s = self.HALF, self.SEARCH
        x0 = max(0, self.x - h - s)
        y0 = max(0, self.y - h - s)
        window = gray[y0:self.y + h + s, x0:self.x + h + s].astype(np.float32)

        match = cv2.matchTemplate(window, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, best = cv2.minMaxLoc(match)
        self.x = x0 + best[0] + h
        self.y = y0 + best[1] + h

        # slowly adapt the template, the plate rotates a bit during the lift
        patch = self._patch(gray).astype(np.float32)
        if patch.shape == self.template.shape and score > 0.6:
            self.template = 0.9 * self.template + 0.1 * patch

        return self.x, self.y, score


def track(path, start, hand):
    """Returns ([x, y, score] or None per frame, plate radius in px)."""
    positions = []
    tracker = None
    radius = None

    for i, gray in enumerate(tqdm(video.frames(path, gray=True), desc="bar", unit="fr")):
        if i < start:
            positions.append(None)
            continue

        if tracker is None:
            x, y, radius = find_plate(gray, hand)
            tracker = HubTracker(gray, x, y)

        positions.append(list(tracker.update(gray)))

    return positions, radius
