"""Shared constants: frame size, colors and layout."""

# output is always a vertical 9:16 video
WIDTH = 1080
HEIGHT = 1920
FPS = 30

# standard bumper plate, used to convert pixels to centimetres
PLATE_DIAMETER_CM = 45.0

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
POSE_MODEL_FILE = "pose_landmarker_heavy.task"

# mediapipe keypoint ids (left side of the body)
EAR = 7
SHOULDER = 11
WRIST = 15
HIP = 23
KNEE = 25
ANKLE = 27
TOES = 31

# colors (RGB)
LIME = (198, 255, 61)
CYAN = (61, 232, 255)
WHITE = (255, 255, 255)
ORANGE = (255, 170, 40)
RED = (255, 80, 70)
GREY = (165, 172, 184)
DIM = (110, 116, 128)
PANEL = (10, 12, 16)

# instagram story safe area (top bar and reply field cover the rest)
SAFE_TOP = 120
SAFE_BOTTOM = 1730

# how long a warning stays on screen
TOAST_FRAMES = 60
