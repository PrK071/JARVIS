from .bridge import fetch_account
from .view import display_name


def render_account(account_id):
    account = fetch_account(account_id)
    return display_name(account)
