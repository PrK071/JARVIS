from .decoder import decode


def forward(payload):
    return decode(raw=payload)
