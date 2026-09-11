from .repository import load_order
from .totals import total


def checkout(order_id):
    order = load_order(order_id)
    return total(order)
