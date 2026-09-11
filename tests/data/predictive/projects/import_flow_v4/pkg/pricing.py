from .catalog import current_item


def current_price():
    return current_item().price
