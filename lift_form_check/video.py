"""Reading and writing video with imageio-ffmpeg."""

import numpy as np
import imageio_ffmpeg

from .config import WIDTH, HEIGHT, FPS


def frames(path, gray=False):
    """Yield frames scaled to 1080x1920 @ 30 fps (RGB or grayscale)."""
    pix_fmt = "gray" if gray else "rgb24"
    reader = imageio_ffmpeg.read_frames(
        path,
        pix_fmt=pix_fmt,
        bits_per_pixel=8 if gray else 24,
        output_params=["-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}"],
    )
    next(reader)  # first item is metadata

    shape = (HEIGHT, WIDTH) if gray else (HEIGHT, WIDTH, 3)
    for raw in reader:
        yield np.frombuffer(raw, np.uint8).reshape(shape)


def writer(path, audio_from=None):
    """Instagram friendly H.264 writer. Send RGB frames with .send(frame)."""
    w = imageio_ffmpeg.write_frames(
        path,
        (WIDTH, HEIGHT),
        fps=FPS,
        codec="libx264",
        quality=None,
        macro_block_size=1,   # keep 1080 wide (default would pad to 1088)
        pix_fmt_out="yuv420p",
        output_params=[
            "-crf", "18",
            "-profile:v", "high",
            "-level", "4.2",
            "-maxrate", "16M",
            "-bufsize", "32M",
            "-movflags", "+faststart",
        ],
        audio_path=audio_from,
        audio_codec="aac",
    )
    w.send(None)  # start the generator
    return w
