from pkg.calc import add_tax


def test_add_tax():
    assert add_tax(10, 2) == 12
