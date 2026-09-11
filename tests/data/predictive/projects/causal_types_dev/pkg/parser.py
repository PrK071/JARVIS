def parse_count(raw):
    return raw


def parse_optional(raw):
    if raw == "":
        return None
    return int(raw)
