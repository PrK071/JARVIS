DEFAULTS = {"timeout": 30}


def timeout(settings):
    value = settings["timeout"]
    if value <= 0:
        raise ValueError("timeout must be positive")
    return value
