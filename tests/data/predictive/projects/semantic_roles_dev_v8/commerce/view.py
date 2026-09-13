from commerce.bridge import relay_invoice, adjusted_amount


def invoice_label():
    return relay_invoice().label


def invoice_amount():
    return adjusted_amount() * 2
