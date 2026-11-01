"""Text layer: header, live stats, warnings and the summary at the end."""

import numpy as np
from PIL import Image, ImageDraw

from . import draw, exercises, i18n
from .config import (WIDTH, HEIGHT, FPS, SAFE_TOP, SAFE_BOTTOM, TOAST_FRAMES,
                     LIME, CYAN, WHITE, ORANGE, RED, GREY, DIM, PANEL)

H1 = draw.font("DejaVuSans-Bold.ttf", 54)
H2 = draw.font("DejaVuSans-Bold.ttf", 28)
TEXT = draw.font("DejaVuSans.ttf", 25)
NUMBER = draw.font("DejaVuSans-Bold.ttf", 48)
MONO = draw.font("DejaVuSansMono.ttf", 19)
MONO_BOLD = draw.font("DejaVuSansMono-Bold.ttf", 24)
TABLE = draw.font("DejaVuSansMono-Bold.ttf", 34)

KIND_COLOR = {"warn": RED, "info": ORANGE, "ok": LIME}
KIND_ICON = {"warn": "!", "info": "i", "ok": "✓"}

STATS_TOP = SAFE_BOTTOM - 235


def header(d, i, profile, load, reps, live):
    top = SAFE_TOP
    draw.panel(d, (36, top, WIDTH - 36, top + 196), outline=(*LIME, 90))

    d.text((70, top + 22), i18n.t("title"), font=H1, fill=WHITE)
    d.text((72, top + 90), f"{profile['title']} · {i18n.t('subtitle')}", font=TEXT, fill=GREY)

    draw.tag(d, (78, top + 152), f"{i18n.t('load')} {load}", MONO_BOLD, LIME)
    draw.tag(d, (452, top + 152), f"SET 1 · {len(reps)} {i18n.t('reps')}", MONO_BOLD, CYAN)

    blink = 255 if (i // 15) % 2 == 0 else 70
    d.ellipse((WIDTH - 190, top + 140, WIDTH - 172, top + 158), fill=(*RED, blink))
    d.text((WIDTH - 162, top + 149), "LIVE", font=MONO_BOLD, fill=WHITE, anchor="lm")

    timecode = f"{i // FPS // 60:02d}:{i // FPS % 60:02d}.{i % FPS:02d}"
    d.text((44, top + 212), f"POSE 33 kp · {FPS} FPS · TC {timecode} · FRM {i:04d}", font=MONO, fill=DIM)

    if live:
        done = sum(i >= r["frame"] for r in reps)
        d.text((WIDTH - 44, top + 212), f"REP {done}/{len(reps)}", font=MONO_BOLD, fill=WHITE, anchor="ra")


def stats(d, i, a, start):
    draw.panel(d, (36, STATS_TOP, WIDTH - 36, SAFE_BOTTOM), outline=(*CYAN, 70))

    values = [
        (i18n.t("hip_angle"), a.hip[i], LIME),
        (i18n.t("torso"), a.torso[i], LIME),
        (i18n.t("knee_angle"), a.knee[i], CYAN),
    ]
    column = (WIDTH - 72) / 3
    for n, (name, value, color) in enumerate(values):
        x = 36 + column * n + column / 2
        d.text((x, STATS_TOP + 16), name, font=MONO, fill=GREY, anchor="ma")
        d.text((x, STATS_TOP + 40), f"{value:.0f}°", font=NUMBER, fill=color, anchor="ma")

    bar_graph(d, i, a, start)


def bar_graph(d, i, a, start, seconds=8):
    """Bar height over the last few seconds with the knee line."""
    x0, x1 = 70, WIDTH - 70
    y0, y1 = STATS_TOP + 135, STATS_TOP + 215
    span = seconds * FPS

    # fixed vertical range around the knee so the graph doesn't rescale
    low, high = a.knee_y - 260, a.knee_y + 80

    def to_screen(k, y):
        return x0 + k / span * (x1 - x0), y0 + (y - low) / (high - low) * (y1 - y0)

    d.text((x0, y0 - 12), i18n.t("bar_path"), font=MONO, fill=GREY, anchor="ls")

    knee_line = to_screen(0, a.knee_y)[1]
    for x in range(x0, x1, 14):
        d.line((x, knee_line, x + 7, knee_line), fill=(*ORANGE, 200), width=2)

    first = max(start, i - span)
    if i <= first + 1:
        return
    line = [to_screen(k, y) for k, y in enumerate(a.hub[first:i + 1, 1])]
    d.line(line, fill=(*CYAN, 255), width=4, joint="curve")
    x, y = line[-1]
    d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=WHITE)


def toast(d, i, toasts):
    """One warning at a time, slides in from the left under the header."""
    for start, kind, title, text in toasts:
        if not start <= i < start + TOAST_FRAMES:
            continue

        slide = draw.ease((i - start) / 7) - draw.ease((i - start - TOAST_FRAMES + 7) / 7)
        x = 36 - (1 - slide) * 1100
        y = SAFE_TOP + 262
        color = KIND_COLOR[kind]

        draw.panel(d, (x, y, x + WIDTH - 72, y + 96), outline=(*color, 220), alpha=200)
        d.ellipse((x + 26, y + 24, x + 72, y + 70), fill=color)
        d.text((x + 49, y + 47), KIND_ICON[kind], font=H2, fill=PANEL, anchor="mm")
        d.text((x + 94, y + 16), title, font=H2, fill=color)
        d.text((x + 94, y + 56), text, font=TEXT, fill=WHITE)


def summary(d, fade, reps, profile):
    d.rectangle((0, SAFE_TOP + 240, WIDTH, HEIGHT), fill=(8, 10, 14, int(215 * fade)))

    y = SAFE_TOP + 280
    d.text((60, y), i18n.t("summary"), font=H1, fill=WHITE)

    columns = profile["columns"]
    xs = [70, 230, 480, 690, 880]

    y += 90
    names = [i18n.t("rep")] + [i18n.pick(label) for _, _, label in columns]
    for x, name in zip(xs, names):
        d.text((x, y), name, font=MONO_BOLD, fill=GREY)

    y += 50
    for n, rep in enumerate(reps, 1):
        warned = {metric for kind, metric, _, _ in rep["flags"] if kind == "warn"}
        d.line((60, y - 12, WIDTH - 60, y - 12), fill=(*DIM, 120))
        d.text((xs[0], y), str(n), font=TABLE, fill=WHITE)

        for x, (key, fmt, _) in zip(xs[1:], columns):
            color = RED if key in warned else LIME
            d.text((x, y), fmt.format(rep[key]), font=TABLE, fill=color)
        y += 70

    y += 30
    for kind, line in exercises.conclusions(reps, profile):
        d.ellipse((64, y + 4, 100, y + 40), fill=KIND_COLOR[kind])
        d.text((82, y + 22), KIND_ICON[kind], font=H2, fill=PANEL, anchor="mm")
        d.text((120, y + 6), line, font=TEXT, fill=WHITE)
        y += 62


def compose(frame, i, labels, arc_label, a, ctx, end_fade=0.0):
    """Puts the text layer over the frame and returns an RGB array."""
    image = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", image.size)
    d = ImageDraw.Draw(layer)

    live = i >= ctx.start
    reps = a.reps

    for text, xy, color in labels:
        draw.tag(d, xy, text, MONO, color)

    header(d, i, ctx.profile, ctx.load, reps, live)

    if ctx.strap and ctx.strap[0] <= i < ctx.strap[1]:
        draw.tag(d, (WIDTH // 2, 1180), f"{i18n.t('strapping')}  ▶▶ {ctx.strap_speed}", MONO_BOLD, WHITE, "mm")

    if live and not end_fade:
        draw.tag(d, arc_label, f"{a.hip[i]:.0f}°", H2, LIME, "mm")
        draw.tag(d, (WIDTH - 64, a.knee_y - 26), i18n.t("knee_line"), MONO, ORANGE, "rm")
        stats(d, i, a, ctx.start)
        toast(d, i, ctx.toasts)

    if end_fade:
        summary(d, end_fade, reps, ctx.profile)

    return np.array(Image.alpha_composite(image, layer).convert("RGB"))
