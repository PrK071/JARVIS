from .client import next_attempt


def handle(base):
    return next_attempt(base)
