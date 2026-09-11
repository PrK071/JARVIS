from app.inventory import reserve, reserve_optional


def test_reserve():
    assert reserve(2) == 1
    assert reserve_optional("2") == 1
