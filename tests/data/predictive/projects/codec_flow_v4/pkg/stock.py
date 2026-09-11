from pkg.decoder import decode_optional, decode_units


def reserve(raw):
    units = decode_units(raw)
    return units - 1


def nested_reserve(raw):
    return apply_reservation(decode_units(raw))


def apply_reservation(units):
    return units - 1


def double_optional(raw):
    units = decode_optional(raw)
    return units * 2
