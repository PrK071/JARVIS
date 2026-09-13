from inventory.summary import remaining


def test_remaining(item):
    assert remaining(item) == 2
