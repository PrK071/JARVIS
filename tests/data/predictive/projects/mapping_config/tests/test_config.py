from pkg.config import timeout


def test_positive_timeout():
    assert timeout({"timeout": 5}) == 5
