from calculator import add


def test_add_positive_integers():
    assert add(2, 3) == 5


def test_add_negative_integer():
    assert add(-2, 3) == 1
