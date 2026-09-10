from .domain import Invoice

def invoice_total(invoice: Invoice) -> int:
    return invoice.subtotal + invoice.tax
