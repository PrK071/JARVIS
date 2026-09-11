from pkg.access import permission
from pkg.accounts import find_account


def test_find_account_returns_record():
    account = find_account([{"id": 7, "label": "ops"}], 7)
    assert account["label"] == "ops"


def test_reader_permission():
    assert permission(False) == "read"
