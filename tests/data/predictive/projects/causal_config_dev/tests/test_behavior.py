from pkg.branch import shipping
from pkg.client import delays


def test_shipping_regular():
    assert shipping(False) == 5


def test_delays():
    assert delays(2) == [0, 1]
