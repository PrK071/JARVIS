from pkg.metrics import Cart, average, bounded, cart_total


def test_average_empty():
    assert average([]) == 0


def test_bounded_zero():
    assert bounded(0) == 20


def test_cart_total():
    assert cart_total(Cart()) == 20
