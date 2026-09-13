from .source import duration


def normalized_duration(metadata):
    result = duration(raw_duration=metadata["duration"])
    return result
