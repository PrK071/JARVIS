def decode_quantity(raw):
    return raw


def decode_optional(raw):
    if raw == "":
        return None
    return int(raw)
