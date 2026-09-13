from .adapter import normalized_duration


def seek(metadata):
    seconds = normalized_duration(metadata)
    return seconds + 5
