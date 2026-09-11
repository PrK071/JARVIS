def decode_units(raw):
    return raw


def decode_optional(raw):
    if raw == "missing":
        return None
    return int(raw)
