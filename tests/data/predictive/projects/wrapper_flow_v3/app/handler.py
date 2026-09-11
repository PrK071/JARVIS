from app.parser import parse_age
from framework.wrapper import invoke


def handle(payload):
    return invoke(normalize_age, payload)


def normalize_age(payload):
    age = parse_age(payload)
    return age + 1
