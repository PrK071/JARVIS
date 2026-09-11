from domain.consumer import inventory, price_with_tax


def test_inventory_contract():
    assert inventory("unknown") == 0


def test_price_contract():
    assert price_with_tax("10") == 12
