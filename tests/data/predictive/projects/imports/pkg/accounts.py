from .sessions import current_session


def current_account():
    return current_session().account
