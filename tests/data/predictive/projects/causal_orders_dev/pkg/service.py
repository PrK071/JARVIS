from pkg.repository import load_order, load_tax
from pkg.totals import calculate


def checkout(order_id):
    order = load_order(order_id)
    return calculate(order)


def checkout_with_tax(order, raw_tax):
    tax = load_tax(raw_tax)
    return calculate_with_tax(order=order, tax=tax)


def calculate_with_tax(order, tax=0):
    return order.total + tax
