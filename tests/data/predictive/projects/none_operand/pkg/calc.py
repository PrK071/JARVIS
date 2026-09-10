def add_tax(subtotal, tax):
    return subtotal + tax


def total_from_order(order):
    return add_tax(order.subtotal, order.tax)
