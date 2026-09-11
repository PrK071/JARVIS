from pkg.service import checkout


def test_missing_order():
    assert checkout("missing") == 0


def test_order_total():
    assert checkout("known") == 15
