from calculator import add


def test_adds_positive_integers() -> None:
    assert add(2, 3) == 5
