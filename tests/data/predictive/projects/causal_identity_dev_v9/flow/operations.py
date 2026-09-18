def divide(total, count):
    return total / count


def average(values):
    return divide(sum(values), len(values))


def scheduled(total, *, slots):
    return divide(total, slots)


def identity(value):
    return value


def caller_supplies_text():
    return identity("bad") + 1


def parse_count(raw: str) -> int:
    return raw


def consume_count(raw):
    count = parse_count(raw)
    return divide(10, count)


def record_source() -> dict:
    return None


def consume_record():
    return record_source()["id"]


def defaulted(count=0):
    return divide(10, count)


def mixed(total, count=1):
    return divide(total, count)


def run_bad_total():
    return scheduled("bad", slots=2)


def run_bad_slots():
    return scheduled(10, slots=0)


def run_bad_mixed():
    return mixed(10, count=0)
