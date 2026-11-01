"""On-screen text in English (default) or Slovenian."""

LANG = "en"

TEXT = {
    "title":     {"en": "KINEMATIC ANALYSIS", "sl": "KINEMATIČNA ANALIZA"},
    "subtitle":  {"en": "computer vision form check", "sl": "ocena drže s strojnim vidom"},
    "load":      {"en": "LOAD", "sl": "OBREMENITEV"},
    "reps":      {"en": "REPS", "sl": "PON."},
    "rep":       {"en": "REP", "sl": "PON."},
    "strapping": {"en": "STRAPPING IN", "sl": "VEZANJE NA DROG"},
    "knee_line": {"en": "KNEE LEVEL", "sl": "NIVO KOLENA"},
    "hip_angle": {"en": "HIP ANGLE", "sl": "KOT V BOKU"},
    "torso":     {"en": "TORSO LEAN", "sl": "NAKLON TRUPA"},
    "knee_angle": {"en": "KNEE ANGLE", "sl": "KOT V KOLENU"},
    "bar_path":  {"en": "BAR PATH · VERTICAL", "sl": "POT DROGA · VERTIKALA"},
    "summary":   {"en": "SET SUMMARY", "sl": "POVZETEK SETA"},
    "clean_reps": {"en": "No warnings on {} of {} reps", "sl": "Brez opozoril pri {} od {} ponovitev"},
    "reps_list": {"en": "Rep {}: {}", "sl": "Pon. {}: {}"},
}


def t(key):
    """Text from the table above."""
    return TEXT[key][LANG]


def pick(value):
    """Values from exercises.yaml can be {en: ..., sl: ...} or a plain string."""
    if isinstance(value, dict):
        return value[LANG]
    return value
