from .source import load_account


def fetch_account(account_id):
    result = load_account(account_id)
    return result
