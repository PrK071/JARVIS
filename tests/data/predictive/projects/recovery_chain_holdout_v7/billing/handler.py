from .tunnel import relay_invoice
from .receipt import receipt_total


def checkout(invoice_key):
    invoice = relay_invoice(invoice_key)
    return receipt_total(invoice)
