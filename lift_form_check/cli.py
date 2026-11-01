"""
Command line entry point.

    python -m lift_form_check lift.mp4 out.mp4 --exercise rdl --start 17 --load "90 % 1RM"
"""

import argparse
import json
import os
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from . import bar, cleanup, exercises, hud, i18n, metrics, overlay, pose, video
from .config import FPS, TOAST_FRAMES, WRIST
from .draw import ease

console = Console()


@dataclass
class Context:
    """Things the HUD needs besides the analysis."""
    profile: dict
    start: int
    load: str
    strap: tuple | None
    strap_speed: str
    toasts: list


def parse_args():
    p = argparse.ArgumentParser(description="Form check for barbell lifts from a side view video.")
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--exercise", choices=list(exercises.PROFILES), default="rdl")
    p.add_argument("--start", type=float, required=True, help="second when the feet are set")
    p.add_argument("--load", default="", help='text for the header, e.g. "120 kg"')
    p.add_argument("--slow", type=float, default=1.0, help="playback speed of the set part, e.g. 0.75")
    p.add_argument("--strap", type=float, nargs=2, metavar=("FROM", "TO"),
                   help="seconds of a sped up part, only shows a label")
    p.add_argument("--strap-speed", default="3×")
    p.add_argument("--lang", choices=["en", "sl"], default="en")
    p.add_argument("--summary", type=float, default=6.0, help="seconds of summary at the end")
    return p.parse_args()


def schedule_toasts(reps):
    """Warnings come one after another, never on top of each other."""
    toasts = []
    for n, rep in enumerate(reps, 1):
        for kind, _, title, text in rep["flags"]:
            start = rep["frame"] - 8
            if toasts:
                start = max(start, toasts[-1][0] + TOAST_FRAMES)
            toasts.append((start, kind, f"{i18n.t('rep')} {n}: {title}", text))
    return toasts


def print_reps(reps, profile):
    table = Table(title=profile["title"])
    table.add_column(i18n.t("rep"))
    for _, _, label in profile["columns"]:
        table.add_column(i18n.pick(label), justify="right")
    table.add_column("")

    for n, rep in enumerate(reps, 1):
        values = [fmt.format(rep[key]) for key, fmt, _ in profile["columns"]]
        notes = ", ".join(title for kind, _, title, _ in rep["flags"] if kind == "warn")
        table.add_row(str(n), *values, f"[red]{notes}[/red]" if notes else "[green]OK[/green]")

    console.print(table)


def render(args, pts, analysis, ctx, plate_radius):
    out = video.writer(args.out, audio_from=args.video)
    total = len(pts)
    last = None

    for i, frame in enumerate(tqdm(video.frames(args.video), total=total, desc="render", unit="fr")):
        if i >= total:
            break
        last = (i, frame)
        drawn, labels, arc = overlay.render(frame, i, pts, analysis, ctx.start, plate_radius)
        out.send(hud.compose(drawn, i, labels, arc, analysis, ctx))

    # freeze the last frame and show the summary on top
    i, frame = last
    drawn, labels, arc = overlay.render(frame, i, pts, analysis, ctx.start, plate_radius)
    for k in range(int(args.summary * FPS)):
        out.send(hud.compose(drawn, i, labels, arc, analysis, ctx, end_fade=ease(k / 12)))

    out.close()


def main():
    args = parse_args()
    i18n.LANG = args.lang
    profile = exercises.get(args.exercise)
    start = int(args.start * FPS)
    base = os.path.splitext(args.video)[0]

    raw_pose = pose.detect(args.video, cache=base + "_pose.json")
    hand = raw_pose[start][WRIST][:2]
    positions, plate_radius = bar.track(args.video, start, hand)

    pts = cleanup.clean(raw_pose, start, lock_knees=profile["lock_knees"])
    hub, score = cleanup.clean_bar(positions)
    analysis = metrics.analyse(pts, hub, score, start, plate_radius, args.slow, profile)

    print_reps(analysis.reps, profile)
    with open(base + "_reps.json", "w") as f:
        json.dump(analysis.reps, f, indent=1, ensure_ascii=False)

    strap = tuple(int(s * FPS) for s in args.strap) if args.strap else None
    ctx = Context(profile, start, args.load, strap, args.strap_speed, schedule_toasts(analysis.reps))

    render(args, pts, analysis, ctx, plate_radius)
    console.print(f"[bold green]done[/bold green] → {args.out}")


if __name__ == "__main__":
    main()
