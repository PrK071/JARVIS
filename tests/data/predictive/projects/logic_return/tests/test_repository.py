from pkg.repository import find_user


def test_find_user_returns_mapping():
    assert find_user([{"id": 1, "name": "Ada"}], 1)["name"] == "Ada"
