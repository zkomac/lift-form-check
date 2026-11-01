"""Body keypoints with MediaPipe Pose Landmarker."""

import json
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from tqdm import tqdm

from . import video
from .config import WIDTH, HEIGHT, FPS, POSE_MODEL_URL, POSE_MODEL_FILE


def _model():
    if not os.path.exists(POSE_MODEL_FILE):
        print("downloading pose model...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_FILE)

    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=POSE_MODEL_FILE),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect(path, cache=None):
    """
    Returns a list with one item per frame:
    33 x [x_px, y_px, visibility] or None if nobody was found.
    Results are cached because this is the slow part.
    """
    if cache and os.path.exists(cache):
        return json.load(open(cache))

    model = _model()
    result = []

    for i, frame in enumerate(tqdm(video.frames(path), desc="pose", unit="fr")):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        found = model.detect_for_video(image, int(i * 1000 / FPS))

        if not found.pose_landmarks:
            result.append(None)
            continue

        points = [[p.x * WIDTH, p.y * HEIGHT, p.visibility] for p in found.pose_landmarks[0]]
        result.append(points)

    if cache:
        json.dump(result, open(cache, "w"))
    return result
