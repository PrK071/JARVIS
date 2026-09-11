from pkg.parser import parse_count, parse_optional


def increment(raw):
    value = parse_count(raw)
    return value + 1


def doubled(raw):
    value = parse_optional(raw)
    return value * 2


def nested(raw):
    return consume(parse_count(raw))


def consume(value):
    return value + 1
