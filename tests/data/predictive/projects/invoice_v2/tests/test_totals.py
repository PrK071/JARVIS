from pkg.domain import Invoice
from pkg.totals import invoice_total

def test_invoice_total_with_tax():
    assert invoice_total(Invoice(100, 20)) == 120
