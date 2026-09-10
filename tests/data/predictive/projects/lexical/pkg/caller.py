from .user_service import normalize_user


def create_user(payload):
    return normalize_user(payload)
