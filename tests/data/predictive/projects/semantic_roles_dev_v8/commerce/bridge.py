from commerce.source import fetch_invoice, fetch_amount


def relay_invoice():
    return fetch_invoice()


def adjusted_amount():
    return fetch_amount() + 1
