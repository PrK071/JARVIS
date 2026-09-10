from .domain import Invoice

def draft_invoice(subtotal: int) -> Invoice:
    return Invoice(subtotal)
