from framework.wrapper import invoke
from .parser import parse_age


def handle(payload):
    return invoke(parse_age, payload)
