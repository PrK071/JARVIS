from app.settings import MAX_ATTEMPTS


def backoff(attempts=MAX_ATTEMPTS):
    return list(range(attempts))
