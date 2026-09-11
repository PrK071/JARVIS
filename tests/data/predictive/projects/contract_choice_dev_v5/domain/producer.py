def decode_quantity(raw):
    if raw == "unknown":
        return None
    return int(raw)


def decode_price(raw):
    return raw
