"""Exercise profiles from exercises.yaml and the warning rules."""

from pathlib import Path

import yaml

from . import i18n

PROFILES = yaml.safe_load((Path(__file__).parent / "exercises.yaml").read_text(encoding="utf-8"))


def get(name):
    return PROFILES[name]


def check(rep, profile):
    """
    Returns a list of (kind, metric, title, text) for one rep.
    kind is "warn" or "ok". Only measured values are used.
    """
    flags = []

    for rule in profile["rules"]:
        value = rep[rule["metric"]]
        too_low = "below" in rule and value < rule["below"]
        too_high = "above" in rule and value > rule["above"]

        if too_low or too_high:
            title = i18n.pick(rule["title"])
            text = i18n.pick(rule["text"]).format(**rep)
            flags.append(("warn", rule["metric"], title, text))

    if not flags:
        flags.append(("ok", None, "OK", i18n.pick(profile["ok"]).format(**rep)))

    return flags


def conclusions(reps, profile, limit=5):
    """Same warning over several reps -> one line, e.g. 'Rep 1, 2: neck extended'."""
    grouped = {}
    for n, rep in enumerate(reps, 1):
        for kind, metric, title, _ in rep["flags"]:
            grouped.setdefault((kind, title), []).append(str(n))

    lines = []
    for (kind, title), numbers in grouped.items():
        if kind == "ok":
            lines.append(("ok", i18n.t("clean_reps").format(len(numbers), len(reps))))
        else:
            lines.append((kind, i18n.t("reps_list").format(", ".join(numbers), title.lower())))

    lines.append(("info", i18n.pick(profile["note"])))
    return lines[:limit]
