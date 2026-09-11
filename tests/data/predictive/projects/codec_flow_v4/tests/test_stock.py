from pkg.stock import double_optional, reserve


def test_numeric_stock():
    assert reserve(3) == 2
    assert double_optional("3") == 6
