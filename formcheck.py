"""
Lift form check
---------------
Takes a side-view video of a barbell lift (RDL, deadlift, squat) and renders a 9:16 video with
pose overlay, bar path, joint angles, per-rep warnings and a set summary.

Usage:
    python formcheck.py input.mp4 output.mp4 --exercise rdl --start 17 --load "90 % 1RM"

--start   second in the video when the feet are set and the set begins
--slow    if the lift part of the video is slowed down (e.g. 0.75), pass it so
          rep tempo is reported in real seconds
"""
import argparse, json, os, subprocess
import numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

W, H, FPS = 1080, 1920, 30
MODEL = "pose_landmarker_heavy.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
PLATE_CM = 45.0  # standard bumper plate diameter, used for px -> cm


# ---------------------------------------------------------------- video io
def read_frames(path, fmt="rgb24"):
    ch = 1 if fmt == "gray" else 3
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={FPS}",
                          "-f", "rawvideo", "-pix_fmt", fmt, "-"], stdout=subprocess.PIPE)
    while True:
        buf = p.stdout.read(W * H * ch)
        if len(buf) < W * H * ch:
            return
        yield np.frombuffer(buf, np.uint8).reshape((H, W, ch) if ch == 3 else (H, W))


# ---------------------------------------------------------------- 1) pose
def run_pose(video, cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    if not os.path.exists(MODEL):
        subprocess.run(["curl", "-sSL", "-o", MODEL, MODEL_URL], check=True)
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    out = []
    for i, fr in enumerate(read_frames(video)):
        r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=fr), int(i * 1000 / FPS))
        out.append([[l.x * W, l.y * H, l.visibility] for l in r.pose_landmarks[0]] if r.pose_landmarks else None)
    json.dump(out, open(cache, "w"))
    return out


# ---------------------------------------------------------------- 2) bar tracking
def find_plate(gray, hand):
    # one Hough pass on the first frame only (way too slow to run on every frame)
    # the bar is in the hands, so take the circle closest to the wrist
    s = cv2.medianBlur(cv2.resize(gray, (W // 2, H // 2)), 5)
    c = cv2.HoughCircles(s, cv2.HOUGH_GRADIENT, dp=1.5, minDist=60, param1=80, param2=35,
                         minRadius=50, maxRadius=110)
    if c is None:
        raise SystemExit("could not find the plate, try a frame where the bar is clearly visible")
    c = c[0] * 2
    x, y, r = min(c, key=lambda k: np.hypot(k[0] - hand[0], k[1] - hand[1]))
    return int(x), int(y), int(r)


def track_bar(video, start, hand):
    """Follow the plate hub with template matching. Returns [x, y, score] per frame."""
    out, tpl, pos, R, S = [], None, None, 36, 90
    radius = 135
    for i, g in enumerate(read_frames(video, "gray")):
        if i < start:
            out.append(None)
            continue
        if tpl is None:
            x, y, radius = find_plate(g, hand)
            # hub = brightest spot near the circle centre (metal sleeve)
            roi = cv2.GaussianBlur(g[y - 60:y + 60, x - 60:x + 60], (15, 15), 0)
            m = cv2.minMaxLoc(roi)[3]
            pos = (x - 60 + m[0], y - 60 + m[1])
            tpl = g[pos[1] - R:pos[1] + R, pos[0] - R:pos[0] + R].astype(np.float32)
        x, y = pos
        x0, y0 = max(0, x - R - S), max(0, y - R - S)
        res = cv2.matchTemplate(g[y0:y + R + S, x0:x + R + S].astype(np.float32), tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        pos = (x0 + loc[0] + R, y0 + loc[1] + R)
        patch = g[pos[1] - R:pos[1] + R, pos[0] - R:pos[0] + R].astype(np.float32)
        if patch.shape == tpl.shape and score > 0.6:
            tpl = 0.9 * tpl + 0.1 * patch  # slowly adapt, plate rotates a bit
        out.append([pos[0], pos[1], float(score)])
    return out, radius


# ---------------------------------------------------------------- 3) cleanup + metrics
def smooth(a, k=7):
    a = np.asarray(a, float)
    pad = np.pad(a, ((k // 2, k // 2),) + ((0, 0),) * (a.ndim - 1), mode="edge")
    return np.array([pad[i:i + k].mean(0) for i in range(len(a))])


def angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(1) / np.linalg.norm(v1, axis=1) / np.linalg.norm(v2, axis=1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def clean_pose(pose, start, lock_knees=False):
    n = len(pose)
    last = next(p for p in pose if p)
    raw = np.array([[l[:2] for l in (p or last)] for p in pose])  # n x 33 x 2
    pts = raw.copy()
    # plates hide knees/ankles a lot, mediapipe then guesses them way off.
    # anything that jumps away from the rolling median gets replaced by the median
    for k in range(33):
        med = median_filter(raw[:, k], size=(21, 1), mode="nearest")
        bad = np.abs(raw[:, k] - med).max(1) > (35 if k >= 23 else 60)
        pts[bad, k] = med[bad]
    # feet don't move during a barbell lift -> lock them once the set starts
    for k in range(27, 33):
        pts[start:, k] = np.median(pts[start:, k], 0)
    out = smooth(pts.reshape(n, -1), 5).reshape(n, 33, 2)
    if lock_knees:
        # RDL: knees only move a few cm, keep a bit of motion around the median
        for k in (25, 26):
            med = np.median(pts[start:, k], 0)
            dev = np.clip(median_filter(pts[start:, k] - med, size=(31, 1), mode="nearest"), -25, 25)
            out[start:, k] = med + smooth(dev, 15)
    return out


# ---------------------------------------------------------------- exercises
# every exercise says: how a rep is counted, what goes in the summary table and
# which warnings it can raise. all warnings come from measured numbers only.

def rules_rdl(r):
    out = []
    if r["depth"] < 1.5:
        out.append(("warn", "depth", "PREMALO GLOBINE", "drog le do kolena · bok potisni dlje nazaj"))
    if r["neck"] > 12:
        out.append(("warn", "neck", "VRAT V EKSTENZIJI", f"+{r['neck']:.0f}° glede na hrbet · pogled 1–2 m pred stopala"))
    if r["down"] < 1.5:
        out.append(("warn", "fast", "PREHITER SPUST", f"spust {r['down']:.1f} s · cilj 2–3 s"))
    return out or [("ok", "ok", "OK", f"drog ~{r['depth']:.0f} cm pod kolenom · vrat nevtralen")]


def rules_squat(r):
    out = []
    if r["depth"] < 0:
        out.append(("warn", "depth", "NAD PARALELO", f"bok {abs(r['depth']):.0f} cm nad kolenom"))
    if r["neck"] > 12:
        out.append(("warn", "neck", "VRAT V EKSTENZIJI", f"+{r['neck']:.0f}° glede na hrbet"))
    if r["down"] < 1.0:
        out.append(("warn", "fast", "NEKONTROLIRAN SPUST", f"spust {r['down']:.1f} s"))
    return out or [("ok", "ok", "OK", f"bok {r['depth']:.0f} cm pod kolenom")]


def rules_deadlift(r):
    out = []
    if r["lockout"] < 165:
        out.append(("warn", "lockout", "NEPOPOLN LOCKOUT", f"kot v boku zgoraj {r['lockout']:.0f}° · stisni zadnjico"))
    if r["hips"] > 1.6:
        out.append(("warn", "hips", "BOKI PREHITRO GOR", f"bok {r['hips']:.1f}× hitreje od droga · potisni s tlemi"))
    if r["neck"] > 12:
        out.append(("warn", "neck", "VRAT V EKSTENZIJI", f"+{r['neck']:.0f}° glede na hrbet"))
    return out or [("ok", "ok", "OK", f"lockout {r['lockout']:.0f}° · bok in drog skupaj")]


def rules_generic(r):
    out = []
    if r["neck"] > 12:
        out.append(("warn", "neck", "VRAT V EKSTENZIJI", f"+{r['neck']:.0f}° glede na hrbet"))
    return out or [("ok", "ok", "OK", "brez opozoril")]


EXERCISES = {
    # count: "bottom" = rep happens at the lowest bar point, "top" = at lockout
    "rdl": dict(title="Romanian Deadlift (RDL)", count="bottom", lock_knees=True, rules=rules_rdl,
                cols=[("GLOBINA", "depth", "{:+.0f} cm"), ("KOT BOK", "hip", "{:.0f}°"),
                      ("SPUST", "down", "{:.1f} s"), ("VRAT", "neck", "{:+.0f}°")],
                note="Globina = cm droga pod kolenom (ocena iz 2D)"),
    "squat": dict(title="Back Squat", count="bottom", lock_knees=False, rules=rules_squat,
                  cols=[("GLOBINA", "depth", "{:+.0f} cm"), ("KOT KOLENO", "knee", "{:.0f}°"),
                        ("SPUST", "down", "{:.1f} s"), ("VRAT", "neck", "{:+.0f}°")],
                  note="Globina = cm boka pod kolenom (ocena iz 2D)"),
    "deadlift": dict(title="Deadlift", count="top", lock_knees=False, rules=rules_deadlift,
                     cols=[("LOCKOUT", "lockout", "{:.0f}°"), ("BOK/DROG", "hips", "{:.1f}×"),
                           ("DVIG", "up", "{:.1f} s"), ("VRAT", "neck", "{:+.0f}°")],
                     note="BOK/DROG = hitrost boka proti drogu na začetku vleka"),
    "generic": dict(title="Analiza giba", count="bottom", lock_knees=False, rules=rules_generic,
                    cols=[("KOT BOK", "hip", "{:.0f}°"), ("KOT KOLENO", "knee", "{:.0f}°"),
                          ("SPUST", "down", "{:.1f} s"), ("VRAT", "neck", "{:+.0f}°")],
                    note="Koti so ocena iz 2D posnetka"),
}


def analyse(pts, bar, start, plate_r, slow, ex):
    first = next(i for i, b in enumerate(bar) if b)
    hub = smooth(np.array([b[:2] if b else bar[first][:2] for b in bar], float), 5)
    conf = np.array([b[2] if b else np.nan for b in bar])
    # left side faces the camera here (11 shoulder, 23 hip, 25 knee, 27 ankle, 7 ear)
    sh, hip, knee, ank, ear = (pts[:, k] for k in (11, 23, 25, 27, 7))

    m = dict(hub=hub, conf=conf)
    m["hip"] = angle(sh, hip, knee)
    m["knee"] = angle(hip, knee, ank)
    d = sh - hip
    m["torso"] = np.degrees(np.arctan2(-d[:, 1], np.abs(d[:, 0])))
    # neck vs back line, positive = head tilted back (extension)
    u, v = sh - hip, ear - sh
    m["neck"] = np.degrees(np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], (u * v).sum(1)))

    dist = int(1.3 * FPS / slow)
    bottoms = [int(b) for b in find_peaks(hub[:, 1], prominence=120, distance=dist)[0] if b > start]
    tops = [int(t) for t in find_peaks(-hub[:, 1], prominence=120, distance=dist)[0] if t > start - 30]
    knee_y = float(np.median(knee[start:, 1]))
    px_cm = 2 * plate_r / PLATE_CM

    reps = []
    if EXERCISES[ex]["count"] == "bottom":
        for b in bottoms:
            top = max([t for t in tops if t < b], default=start)
            depth = (hub[b][1] - knee_y) if ex == "rdl" else (hip[b][1] - knee[b][1])
            reps.append(dict(frame=b, depth=depth / px_cm, hip=m["hip"][b], knee=m["knee"][b],
                             neck=m["neck"][b], down=(b - top) / FPS * slow))
    else:  # deadlift: one rep = floor -> lockout
        for t in tops:
            if t <= start:
                continue
            b = max([x for x in bottoms if x < t], default=start)
            # first 40 % of the pull: how fast do hips rise compared to the bar?
            e = b + max(2, int((t - b) * 0.4))
            ratio = (hip[b, 1] - hip[e, 1]) / max(hub[b, 1] - hub[e, 1], 1)
            w = slice(max(t - 5, 0), t + 6)
            reps.append(dict(frame=t, lockout=float(m["hip"][w].max()), hips=float(ratio),
                             neck=m["neck"][b], up=(t - b) / FPS * slow, hip=m["hip"][t]))
    for r in reps:
        r["flags"] = EXERCISES[ex]["rules"](r)
    m.update(reps=reps, knee_y=knee_y,
             midfoot=float(np.median(((ank[:, 0] + pts[:, 31, 0]) / 2)[start:])))
    return m


def conclusions(reps, ex):
    """Group the same warning over reps -> 'Pon. 1, 2: ...'"""
    seen = {}
    for n, r in enumerate(reps, 1):
        for kind, code, title, _ in r["flags"]:
            seen.setdefault((kind, code, title), []).append(str(n))
    lines = []
    for (kind, code, title), ns in seen.items():
        if code == "ok":
            lines.append(("ok", f"Brez opozoril pri {len(ns)}/{len(reps)} ponovitvah"))
        else:
            lines.append((kind, f"Pon. {', '.join(ns)}: {title.lower()}"))
    lines.append(("info", EXERCISES[ex]["note"]))
    return lines[:5]


# ---------------------------------------------------------------- 4) drawing
LIME, CYAN, WHITE = (198, 255, 61), (61, 232, 255), (255, 255, 255)
ORANGE, RED, GREY, DIM = (255, 170, 40), (255, 80, 70), (165, 172, 184), (110, 116, 128)
COL = dict(warn=RED, info=ORANGE, ok=LIME)
ICON = dict(warn="!", info="i", ok="✓")
TOP, BOTTOM = 120, 1730          # keep text inside instagram story safe area
TOAST = 60                       # frames a warning stays on screen
FD = "/usr/share/fonts/truetype/dejavu/"
font = lambda name, size: ImageFont.truetype(FD + name, size)
F_H1, F_H2, F_TXT = font("DejaVuSans-Bold.ttf", 54), font("DejaVuSans-Bold.ttf", 28), font("DejaVuSans.ttf", 25)
F_NUM, F_MONO, F_MONOB = font("DejaVuSans-Bold.ttf", 48), font("DejaVuSansMono.ttf", 19), font("DejaVuSansMono-Bold.ttf", 24)
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25),
            (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32), (7, 11), (8, 12)]


def rgb(c):
    return tuple(c)


def ease(t):
    t = min(max(t, 0), 1)
    return 1 - (1 - t) ** 3


def dashed(img, a, b, col, th=2, dash=16, gap=10):
    a, b = np.array(a, float), np.array(b, float)
    length = np.linalg.norm(b - a)
    u = (b - a) / max(length, 1e-6)
    s = 0.0
    while s < length:
        cv2.line(img, tuple(map(int, a + u * s)), tuple(map(int, a + u * min(s + dash, length))), rgb(col), th, cv2.LINE_AA)
        s += dash + gap


def corners(img, x0, y0, x1, y1, col, ln=34, th=3):
    for x, y, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (int(x), int(y)), (int(x + dx * ln), int(y)), rgb(col), th, cv2.LINE_AA)
        cv2.line(img, (int(x), int(y)), (int(x), int(y + dy * ln)), rgb(col), th, cv2.LINE_AA)


def tag(dr, xy, txt, f, col, anchor="lm"):
    bb = dr.textbbox(xy, txt, font=f, anchor=anchor)
    dr.rectangle((bb[0] - 8, bb[1] - 5, bb[2] + 8, bb[3] + 5), fill=(*col, 230))
    dr.text(xy, txt, font=f, fill=(10, 12, 16), anchor=anchor)


def box(dr, xy, outline=None, a=165):
    dr.rounded_rectangle(xy, radius=20, fill=(10, 12, 16, a), outline=outline, width=2 if outline else 0)


def draw_frame(fr, i, pts, m, toasts, args, end=0.0):
    start, reps = args.start_frame, m["reps"]
    live = i >= start
    fade = ease((i - start) / 20)
    lay, glow = np.zeros_like(fr), np.zeros_like(fr)
    kp = pts[i]

    # faint full body mesh + person box
    for a, b in SKELETON:
        cv2.line(lay, tuple(map(int, kp[a])), tuple(map(int, kp[b])), (200, 200, 200), 1, cv2.LINE_AA)
    for p in kp[[0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]]:
        cv2.circle(lay, tuple(map(int, p)), 4, (230, 230, 230), -1, cv2.LINE_AA)
    x0, y0, x1, y1 = kp[:, 0].min() - 40, kp[:, 1].min() - 60, kp[:, 0].max() + 40, kp[:, 1].max() + 30
    corners(lay, x0, y0, x1, y1, LIME)
    labels = [("PERSON 0.98 · ID 01", (x0, y0 - 14), LIME)]

    arc_label = None
    if live:
        sh, hp, kn, an, ear = kp[11], kp[23], kp[25], kp[27], kp[7]
        dashed(lay, (m["midfoot"], TOP + 380), (m["midfoot"], BOTTOM - 255), GREY)
        dashed(lay, (60, m["knee_y"]), (W - 60, m["knee_y"]), ORANGE, 2, 12, 9)
        u = (sh - hp) / np.linalg.norm(sh - hp)
        neck_col = RED if m["neck"][i] > 12 else (230, 230, 230)
        for a, b, c, t in ((hp - u * 50, sh + u * 100, LIME, 6), (hp, kn, CYAN, 5), (kn, an, CYAN, 5), (sh, ear, neck_col, 4)):
            cv2.line(glow, tuple(map(int, a)), tuple(map(int, b)), rgb(c), t * 4, cv2.LINE_AA)
            cv2.line(lay, tuple(map(int, a)), tuple(map(int, b)), rgb(c), t, cv2.LINE_AA)
        for p, c, r in ((ear, neck_col, 9), (sh, LIME, 12), (hp, LIME, 14), (kn, CYAN, 12), (an, CYAN, 10)):
            cv2.circle(lay, tuple(map(int, p)), r + 5, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(lay, tuple(map(int, p)), r, rgb(c), -1, cv2.LINE_AA)
        # hip angle arc
        a1 = np.degrees(np.arctan2(*(sh - hp)[::-1]))
        a2 = np.degrees(np.arctan2(*(kn - hp)[::-1]))
        dl = (a2 - a1 + 540) % 360 - 180
        cv2.ellipse(lay, tuple(map(int, hp)), (66, 66), 0, a1, a1 + dl, rgb(LIME), 4, cv2.LINE_AA)
        mid = np.radians(a1 + dl / 2)
        arc_label = (hp[0] + np.cos(mid) * 118, hp[1] + np.sin(mid) * 118)
        # bar path trail (last 2.5 s)
        trail = m["hub"][max(start, i - 75):i + 1]
        for j in range(1, len(trail)):
            c = tuple(int(v * j / len(trail)) for v in rgb(CYAN))
            cv2.line(glow, tuple(map(int, trail[j - 1])), tuple(map(int, trail[j])), c, 14, cv2.LINE_AA)
            cv2.line(lay, tuple(map(int, trail[j - 1])), tuple(map(int, trail[j])), c, 5, cv2.LINE_AA)
        h, R = m["hub"][i], args.plate_r
        cv2.rectangle(lay, (int(h[0] - R), int(h[1] - R)), (int(h[0] + R), int(h[1] + R)), rgb(CYAN), 1, cv2.LINE_AA)
        corners(lay, h[0] - R, h[1] - R, h[0] + R, h[1] + R, CYAN, 26)
        cv2.drawMarker(lay, tuple(map(int, h)), rgb(WHITE), cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
        score = m["conf"][i] if not np.isnan(m["conf"][i]) else 0.9
        labels.append((f"BARBELL {score:.2f} · TRK", (h[0] - R, h[1] + R + 22), CYAN))

    # blend overlay on top of the (slightly darkened) frame
    glow = cv2.GaussianBlur(glow, (0, 0), 9)
    k = fade if live else 0
    fr = (fr * (1 - 0.22 * max(k, 0.4))).astype(np.uint8)
    fr = cv2.add(fr, (glow * 0.9 * k).astype(np.uint8))
    alpha = np.where(lay.any(2), 0.85, 0)[..., None]
    fr = (fr * (1 - alpha) + lay * alpha).astype(np.uint8)

    img = Image.fromarray(fr).convert("RGBA")
    ov = Image.new("RGBA", img.size)
    dr = ImageDraw.Draw(ov)
    for txt, xy, c in labels:
        tag(dr, xy, txt, F_MONO, c)

    # header
    box(dr, (36, TOP, W - 36, TOP + 196), outline=(*LIME, 90))
    dr.text((70, TOP + 22), "KINEMATIČNA ANALIZA", font=F_H1, fill=WHITE)
    dr.text((72, TOP + 90), f"{EXERCISES[args.exercise]['title']} · ocena drže s strojnim vidom", font=F_TXT, fill=GREY)
    tag(dr, (78, TOP + 152), f"OBREMENITEV {args.load}", F_MONOB, LIME)
    tag(dr, (452, TOP + 152), f"SET 1 · {len(reps)} PON.", F_MONOB, CYAN)
    dr.ellipse((W - 190, TOP + 140, W - 172, TOP + 158), fill=(*RED, 255 if (i // 15) % 2 == 0 else 70))
    dr.text((W - 162, TOP + 149), "LIVE", font=F_MONOB, fill=WHITE, anchor="lm")
    tc = f"{i // FPS // 60:02d}:{i // FPS % 60:02d}.{i % FPS:02d}"
    dr.text((44, TOP + 212), f"POSE 33 kp · {FPS} FPS · TC {tc} · FRM {i:04d}", font=F_MONO, fill=DIM)
    if live:
        dr.text((W - 44, TOP + 212), f"REP {sum(i >= r['frame'] for r in reps)}/{len(reps)}", font=F_MONOB, fill=WHITE, anchor="ra")

    if args.strap and args.strap[0] <= i < args.strap[1]:
        tag(dr, (W // 2, 1180), f"VEZANJE NA DROG  ▶▶ {args.strap_speed}", F_MONOB, WHITE, anchor="mm")

    if live and not end:
        tag(dr, arc_label, f"{m['hip'][i]:.0f}°", F_H2, LIME, anchor="mm")
        tag(dr, (W - 64, m["knee_y"] - 26), "NIVO KOLENA", F_MONO, ORANGE, anchor="rm")
        draw_stats(dr, i, m, start)
        draw_toast(dr, i, toasts)
    if end:
        draw_summary(dr, end, reps, args)
    return np.array(Image.alpha_composite(img, ov).convert("RGB"))


def draw_stats(dr, i, m, start):
    y0 = BOTTOM - 235
    box(dr, (36, y0, W - 36, BOTTOM), outline=(*CYAN, 70))
    cw = (W - 72) / 3
    for k, (name, val, c) in enumerate([("KOT V BOKU", m["hip"][i], LIME), ("NAKLON TRUPA", m["torso"][i], LIME),
                                         ("KOT V KOLENU", m["knee"][i], CYAN)]):
        cx = 36 + cw * k + cw / 2
        dr.text((cx, y0 + 16), name, font=F_MONO, fill=GREY, anchor="ma")
        dr.text((cx, y0 + 40), f"{val:.0f}°", font=F_NUM, fill=c, anchor="ma")
    # small graph of bar height, last 8 s
    gx0, gx1, gy0, gy1 = 70, W - 70, y0 + 135, y0 + 215
    lo, hi = 780, 1120
    sy = lambda y: gy0 + (y - lo) / (hi - lo) * (gy1 - gy0)
    dr.text((gx0, gy0 - 12), "POT DROGA · VERTIKALA", font=F_MONO, fill=GREY, anchor="ls")
    for x in range(gx0, gx1, 14):
        dr.line((x, sy(m["knee_y"]), x + 7, sy(m["knee_y"])), fill=(*ORANGE, 200), width=2)
    s = max(start, i - 240)
    if i > s + 1:
        p = [(gx0 + k / 240 * (gx1 - gx0), sy(y)) for k, y in enumerate(m["hub"][s:i + 1, 1])]
        dr.line(p, fill=(*CYAN, 255), width=4, joint="curve")
        dr.ellipse((p[-1][0] - 7, p[-1][1] - 7, p[-1][0] + 7, p[-1][1] + 7), fill=WHITE)


def draw_toast(dr, i, toasts):
    for s, kind, title, sub in toasts:
        if not s <= i < s + TOAST:
            continue
        slide = ease((i - s) / 7) - ease((i - s - TOAST + 7) / 7)
        x, y, c = 36 - (1 - slide) * 1100, TOP + 262, COL[kind]
        box(dr, (x, y, x + W - 72, y + 96), outline=(*c, 220), a=200)
        dr.ellipse((x + 26, y + 24, x + 72, y + 70), fill=c)
        dr.text((x + 49, y + 47), ICON[kind], font=F_H2, fill=(10, 12, 16), anchor="mm")
        dr.text((x + 94, y + 16), title, font=F_H2, fill=c)
        dr.text((x + 94, y + 56), sub, font=F_TXT, fill=WHITE)


FLAG_COL = dict(depth="depth", neck="neck", fast="down", lockout="lockout", hips="hips")


def draw_summary(dr, k, reps, args):
    cols = EXERCISES[args.exercise]["cols"]
    dr.rectangle((0, TOP + 240, W, H), fill=(8, 10, 14, int(215 * k)))
    y = TOP + 280
    dr.text((60, y), "POVZETEK SETA", font=F_H1, fill=WHITE)
    y += 90
    xs = [70, 230, 480, 690, 880]
    for x, t in zip(xs, ["PON."] + [c[0] for c in cols]):
        dr.text((x, y), t, font=F_MONOB, fill=GREY)
    y += 50
    big = font("DejaVuSansMono-Bold.ttf", 34)
    for n, r in enumerate(reps, 1):
        bad = {FLAG_COL.get(f[1]) for f in r["flags"] if f[0] == "warn"}
        row = [(str(n), WHITE)] + [(fmt.format(r[key]), RED if key in bad else LIME) for _, key, fmt in cols]
        dr.line((60, y - 12, W - 60, y - 12), fill=(*DIM, 120))
        for x, (t, c) in zip(xs, row):
            dr.text((x, y), t, font=big, fill=c)
        y += 70
    y += 30
    for kind, t in conclusions(reps, args.exercise):
        dr.ellipse((64, y + 4, 100, y + 40), fill=COL[kind])
        dr.text((82, y + 22), ICON[kind], font=F_H2, fill=(10, 12, 16), anchor="mm")
        dr.text((120, y + 6), t, font=F_TXT, fill=WHITE)
        y += 62


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--exercise", choices=list(EXERCISES), default="rdl")
    ap.add_argument("--start", type=float, required=True, help="second when feet are set")
    ap.add_argument("--load", default="")
    ap.add_argument("--slow", type=float, default=1.0)
    ap.add_argument("--strap", type=float, nargs=2, help="start/end second of sped-up strap part (label only)")
    ap.add_argument("--strap-speed", default="3×")
    ap.add_argument("--summary", type=float, default=6.0, help="seconds of summary at the end")
    args = ap.parse_args()
    args.start_frame = int(args.start * FPS)
    args.strap = [int(s * FPS) for s in args.strap] if args.strap else None

    base = os.path.splitext(args.video)[0]
    print("pose...")
    pose = run_pose(args.video, base + "_pose.json")
    print("bar...")
    bar, args.plate_r = track_bar(args.video, args.start_frame, pose[args.start_frame][15][:2])
    ex = EXERCISES[args.exercise]
    pts = clean_pose(pose, args.start_frame, ex["lock_knees"])
    m = analyse(pts, bar, args.start_frame, args.plate_r, args.slow, args.exercise)
    for n, r in enumerate(m["reps"], 1):
        vals = "  ".join(f"{name.lower()} {fmt.format(r[key])}" for name, key, fmt in ex["cols"])
        print(f"rep {n}: {vals}")
    json.dump([{k: (float(v) if k != "flags" else v) for k, v in r.items()} for r in m["reps"]],
              open(base + "_reps.json", "w"), indent=1)

    # warnings go one after another so they don't overlap
    toasts = []
    for n, r in enumerate(m["reps"], 1):
        for kind, _, title, sub in r["flags"]:
            t = max(r["frame"] - 8, toasts[-1][0] + TOAST if toasts else 0)
            toasts.append((t, kind, f"PON. {n}: {title}", sub))

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-i", args.video, "-map", "0:v", "-map", "1:a?",
                            "-af", "apad", "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                            "-profile:v", "high", "-level", "4.2", "-maxrate", "16M", "-bufsize", "32M", "-g", "60",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
                            "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    last = None
    for i, fr in enumerate(read_frames(args.video)):
        if i >= len(pts):
            break
        last = fr
        enc.stdin.write(draw_frame(fr.copy(), i, pts, m, toasts, args).tobytes())
    for k in range(int(args.summary * FPS)):  # freeze last frame + summary
        enc.stdin.write(draw_frame(last.copy(), i, pts, m, toasts, args, end=ease(k / 12)).tobytes())
    enc.stdin.close()
    enc.wait()
    print("done ->", args.out)


if __name__ == "__main__":
    main()
