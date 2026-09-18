def quotient(amount, units):
    return amount / units


def density(samples):
    return quotient(sum(samples), len(samples))


def allocate(amount, *, buckets):
    return quotient(amount, buckets)


def echo(payload):
    return payload


def submit_bad_payload():
    return echo("invalid") + 2


def decode_units(raw: str) -> int:
    return raw


def calculate_units(raw):
    units = decode_units(raw)
    return quotient(12, units)


def fetch_asset() -> dict:
    return None


def asset_name():
    return fetch_asset()["name"]


def default_units(units=0):
    return quotient(12, units)


def combine(amount, units=1):
    return quotient(amount, units)


def run_bad_amount():
    return allocate("bad", buckets=3)


def run_bad_buckets():
    return allocate(12, buckets=0)


def run_bad_combine():
    return combine(12, units=0)
