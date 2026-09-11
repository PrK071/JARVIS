from app.decoder import decode_optional, decode_quantity


def reserve(raw):
    quantity = decode_quantity(raw)
    return quantity - 1


def reserve_optional(raw):
    return apply_quantity(decode_optional(raw))


def apply_quantity(quantity):
    return quantity - 1
