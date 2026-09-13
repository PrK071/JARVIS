from .origin import obtain_invoice


def relay_invoice(invoice_key):
    invoice = obtain_invoice(invoice_key)
    return invoice
