from .config import DEFAULTS, timeout


def load(raw):
    merged = {**DEFAULTS, **raw}
    return timeout(merged)
