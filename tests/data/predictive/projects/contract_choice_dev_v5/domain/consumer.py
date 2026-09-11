from .producer import decode_price, decode_quantity


def inventory(raw):
    quantity = decode_quantity(raw)
    return quantity + 1


def price_with_tax(raw):
    price = decode_price(raw)
    return price + 2
