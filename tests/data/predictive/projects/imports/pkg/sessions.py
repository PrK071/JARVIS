from .accounts import current_account


def current_session():
    return current_account().session
