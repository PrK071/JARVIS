from pkg.service import checkout


def test_checkout_requires_order():
    assert checkout("missing") == 0
