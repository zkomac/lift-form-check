# lift-form-check

Computer-vision form check for barbell lifts from a single phone video.
Pose estimation + bar tracking → joint angles, depth, tempo and per-rep warnings,
rendered as a 9:16 video ready for an Instagram story/reel.

Supported: **RDL**, **deadlift**, **back squat** (+ a `generic` mode with angles only).

![preview](docs/preview.jpg)

## How it works

![pipeline](docs/pipeline.png)

1. **Pose** – MediaPipe Pose Landmarker (heavy), 33 keypoints per frame.
2. **Cleanup** – plates hide knees and ankles, so the raw detection jumps around a lot.
   A Hampel filter replaces outliers with the rolling median, feet are locked once the set
   starts (they don't move in a barbell lift) and for RDL knees are limited to a small range.
3. **Bar** – the plate is found once with a Hough circle closest to the hands, then the
   plate hub is followed with template matching (a lot faster than running Hough on every frame).
   Plate diameter (45 cm) is used to convert pixels to centimetres.
4. **Metrics** – hip / knee angle, torso lean, neck vs back line, reps from bar height peaks,
   depth vs knee level, descent time.
5. **Render** – overlay with skeleton, bar path, angle arcs, live stats, a bar-height graph,
   warnings that slide in after each rep and a summary table at the end.

![cleanup](docs/cleanup.png)

## Example output (RDL, 5 reps)

![bar path](docs/bar_path.png)
![reps](docs/reps.png)

Still frame analysis:

<img src="docs/still_analysis.png" width="520">

## Warnings

Only things that are actually measured, per exercise:

| Exercise | Warning | Rule |
|---|---|---|
| RDL | not enough depth | bar < 1.5 cm below knee level |
| RDL | descent too fast | < 1.5 s |
| Squat | above parallel | hip above knee at the bottom |
| Squat | uncontrolled descent | < 1.0 s |
| Deadlift | soft lockout | hip angle at the top < 165° |
| Deadlift | hips shoot up | hips rise > 1.6× faster than the bar in the first 40 % of the pull |
| all | neck extension | head > 12° above the back line at the bottom |

## Usage

```bash
pip install -r requirements.txt     # + ffmpeg
python formcheck.py lift.mp4 out.mp4 --exercise rdl --start 17 --load "90 % 1RM"
```

| option | |
|---|---|
| `--exercise` | `rdl`, `deadlift`, `squat`, `generic` |
| `--start` | second when the feet are set (bar out of the rack) |
| `--load` | text for the header, e.g. `"120 kg"` |
| `--slow` | if the set part of the video is slowed down (e.g. `0.75`), tempo is still reported in real seconds |
| `--strap a b` | show a "sped up" label between seconds a and b |

The pose model is downloaded on the first run. Pose results are cached next to the video
(`*_pose.json`), per-rep numbers are saved to `*_reps.json`.
Charts: `python make_figures.py lift.mp4 <start> <slow> <exercise>`.

### Filming tips
- side view, camera at hip height, ~3 m away, whole body and plate in frame
- vertical 1080×1920, 30 fps
- the side of the body facing the camera is used (left side by default)

## Limitations

- 2D only. Angles depend on the camera being perpendicular to the lifter.
- Knees and ankles are behind the plate, so depth is an estimate within a few cm.
- MediaPipe keypoints are not anatomical landmarks (hip ≠ greater trochanter).
- Squat and deadlift rules are simple heuristics, tuned less than RDL.
- On-screen text is in Slovenian.

## Author

Žiga Komac
