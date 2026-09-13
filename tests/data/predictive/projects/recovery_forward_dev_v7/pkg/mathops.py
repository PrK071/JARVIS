from .relay import forward


def doubled(payload):
    value = forward(payload)
    return value * 2
