from .settings import RETRY_LIMIT

def retry_delays() -> list[int]:
    return list(range(RETRY_LIMIT))
