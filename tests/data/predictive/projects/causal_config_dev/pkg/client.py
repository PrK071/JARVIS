from pkg.settings import RETRY_LIMIT


def delays(limit=RETRY_LIMIT):
    return list(range(limit))
