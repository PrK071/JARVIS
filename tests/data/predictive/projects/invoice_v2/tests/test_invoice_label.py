from pkg.domain import Invoice

def test_invoice_keeps_subtotal():
    assert Invoice(100).subtotal == 100
