def parse_age(payload):
    raw_age = payload.get("age")
    return int(raw_age)
