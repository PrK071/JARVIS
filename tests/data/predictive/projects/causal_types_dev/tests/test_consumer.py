from pkg.consumer import doubled, increment, nested


def test_numbers():
    assert increment(2) == 3
    assert doubled("2") == 4
    assert nested(2) == 3
