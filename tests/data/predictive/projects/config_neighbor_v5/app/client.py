from . import settings


def next_attempt(base):
    return base + settings.RETRY_DELAY
