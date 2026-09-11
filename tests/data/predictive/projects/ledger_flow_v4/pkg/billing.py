from pkg.store import fetch_fee, fetch_invoice
from pkg.summary import invoice_total


def bill(invoice_id):
    invoice = fetch_invoice(invoice_id)
    return invoice_total(invoice)


def bill_with_fee(invoice, raw_fee):
    fee = fetch_fee(raw_fee)
    return invoice.amount + fee
