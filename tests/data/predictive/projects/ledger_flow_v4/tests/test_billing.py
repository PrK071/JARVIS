from pkg.billing import bill


def test_missing_invoice_has_no_total():
    assert bill("absent") == 0
